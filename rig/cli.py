"""Local, recorded UniRig pipeline. Heavy inference runs in the module-owned venv."""
import argparse
import contextlib
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from fetch_assets import sha256
from validate_contract import load_contract
from chat_service import chat_state, stop_chat, start_chat
from source_integrity import verify_source


MOD = Path(__file__).resolve().parent
RESULT = {}


class CheckFailure(RuntimeError):
    pass


def parser():
    p = argparse.ArgumentParser(description="Rig a fictional adult humanoid locally; produce checked FBX, GLB and a record")
    p.add_argument("input", type=Path)
    p.add_argument("--json", action="store_true", help="emit one producer JSON result; progress goes to stderr")
    p.add_argument("--name")
    p.add_argument("--triangles", type=int)
    p.add_argument("--height", type=float, default=1.7)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--smooth-iterations", type=int)
    p.add_argument("--facing", choices=("-y", "+x", "+y", "-x"), default="-y",
                   help="Declared input facing in Blender coordinates; verified against the limited UniRig heel reference")
    return p


@contextmanager
def gpu_lease(on_release=None):
    import fcntl
    with open("/tmp/ai-gpu.lock", "a+") as lock:
        deadline = time.monotonic() + 900
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError("GPU remained busy for 15 minutes")
                print("Waiting for the shared GPU lease...", flush=True)
                time.sleep(5)
        os.set_inheritable(lock.fileno(), True)
        try:
            yield lock.fileno()
        finally:
            if on_release:
                on_release()


class VramMonitor:
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        executable = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        while not self.stop.is_set():
            try:
                r = subprocess.run([executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5, check=True)
                self.peak = max(self.peak, max(int(x.strip()) for x in r.stdout.splitlines()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(1)


def execute(args):
    contract = load_contract()
    profile = json.loads((MOD / "profile.json").read_text())
    models = json.loads((MOD / "models.json").read_text())
    installed = json.loads((MOD / "app/installed.json").read_text())
    source = Path((MOD / "app/source-path.txt").read_text().strip())
    source_revision = models["source"]["revision"]
    source_tree_sha256 = verify_source(source, source_revision)
    if len(installed["checkpoints"]) != len(models["weights"]["files"]):
        raise RuntimeError("Installed checkpoint list is incomplete")
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file() or input_path.suffix.lower() != ".glb":
        raise ValueError("Expected an existing input .glb")
    triangles = contract["mesh"]["default_triangle_budget"] if args.triangles is None else args.triangles
    smooth = profile["smooth_iterations"] if args.smooth_iterations is None else args.smooth_iterations
    if not 1000 <= triangles <= contract["mesh"]["max_triangles"] or not 1 <= smooth <= 1000:
        raise ValueError("Triangle/smoothing setting is outside the contract")
    if not .3 <= args.height <= 3 or not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("Height or seed is out of range")
    name = args.name or re.sub(r"[^A-Za-z0-9_-]", "_", input_path.stem)[:64] or "character"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Name must contain only letters, digits, underscores or hyphens")
    work = Path(os.environ.get("RIG_WORK_DIR", os.environ.get("WORK_DIR", str(Path.home() / "work")))).expanduser().resolve()
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = work / "outputs/rig" / (name + "-" + job_id)
    temporary = work / "tmp/rig" / job_id
    output.mkdir(parents=True, exist_ok=False)
    temporary.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"; logs.mkdir()
    record_file = output / "record.json"
    record = dict(schema_version=1, job_id=job_id, pid=os.getpid(), status="preparing",
                  contract=contract["id"], contract_sha256=sha256(MOD/"contract.json"),
                  source_revision=source_revision, input_sha256=sha256(input_path),
                  seed=args.seed, settings=dict(triangles=triangles, height_m=args.height,
                  smooth_iterations=smooth, profile=profile["id"], input_facing=args.facing), stages=[], outputs={},
                  unreal_animation_validation="not_run", peak_vram_mib=0)
    record["runtime_versions"] = {name: importlib.metadata.version(name) for name in
        ("torch", "torchvision", "bpy", "numpy", "flash-attn", "transformers", "spconv-cu121")}
    record["module_code_sha256"] = {p.name: sha256(p) for p in MOD.glob("*.py")}
    record["profile_sha256"] = sha256(MOD/"profile.json")
    record["source_tree_sha256"] = source_tree_sha256
    record["models_manifest_sha256"] = sha256(MOD/"models.json")
    record["dependency_lock_sha256"] = sha256(MOD/"requirements.lock")
    record["checkpoint_sha256"] = {row["path"]: row["sha256"] for row in models["weights"]["files"]}

    def save():
        pending = record_file.with_suffix(".json.new")
        pending.write_text(json.dumps(record, indent=2))
        pending.replace(record_file)

    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", WANDB_MODE="disabled", PYTHONPATH=str(source),
               PYTHONHASHSEED=str(args.seed), TMPDIR=str(temporary), XDG_CACHE_HOME=str(temporary/"cache"),
               TRITON_CACHE_DIR=str(temporary/"cache/triton"), TORCH_HOME=str(temporary/"cache/torch"),
               BLENDER_USER_CONFIG=str(temporary/"blender-config"))
    (temporary/"cache").mkdir()
    lease_fd = None

    def step(label, command, expected=None):
        event = dict(name=label, status="running", started_utc=datetime.now(timezone.utc).isoformat())
        record["stages"].append(event); save()
        print("Rig stage:", label, flush=True)
        started = time.monotonic()
        with (logs/(label+".log")).open("w") as log:
            child = subprocess.Popen([str(x) for x in command], cwd=temporary, env=env,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                pass_fds=(() if lease_fd is None else (lease_fd,)))
            event["child_pid"] = child.pid; save()
            try:
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print("  still running:", label, flush=True)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                event.update(status="interrupted", exit_code=child.returncode,
                             seconds=round(time.monotonic()-started, 2)); save()
                raise
        event.update(exit_code=code, seconds=round(time.monotonic()-started, 2))
        if code or (expected is not None and (not Path(expected).is_file() or Path(expected).stat().st_size == 0)):
            event["status"] = "failed"; save()
            failure = CheckFailure if label in ("finish", "verify_fbx", "verify_glb") else RuntimeError
            raise failure(label + " failed; see " + str(logs/(label+".log")))
        event["status"] = "succeeded"; save()

    save()
    monitor = VramMonitor()
    was_active = False

    def restore_chat():
        # Called before the lease closes, including on failure/cancellation.
        monitor.stop.set()
        if monitor.thread.is_alive(): monitor.thread.join(timeout=6)
        if was_active:
            record["chat_restored"] = False
            start_chat()
            record["chat_restored"] = True

    try:
        record["status"] = "verifying_assets"; save()
        for item, pin in zip(installed["checkpoints"], models["weights"]["files"]):
            if sha256(item["path"]) != pin["sha256"]:
                raise RuntimeError("Pinned checkpoint checksum mismatch")
        architecture = Path(installed["architecture"]["directory"])
        architecture_pin = json.loads((MOD/"architecture.json").read_text())["sha256"]
        if sha256(architecture/"config.json") != architecture_pin:
            raise RuntimeError("Architecture configuration drifted")
        shutil.copytree(source/"configs", temporary/"configs")
        # Pin job configs to verified local assets; upstream's download adapter is bypassed.
        import yaml
        for task, checkpoint in zip((profile["tasks"]["skeleton"], profile["tasks"]["skin"]), installed["checkpoints"]):
            path = temporary/"configs/task"/task
            data = yaml.safe_load(path.read_text()); data["resume_from_checkpoint"] = checkpoint["path"]
            path.write_text(yaml.safe_dump(data))
        path = temporary/"configs/model"/profile["model_config"]
        data = yaml.safe_load(path.read_text()); data["llm"]["pretrained_model_name_or_path"] = str(architecture)
        path.write_text(yaml.safe_dump(data))
        blender = shutil.which("blender")
        if not blender: raise RuntimeError("Blender module is not available")
        prepared = temporary/"prepared.glb"
        # CPU-only preprocessing uses the installed executable rather than unloading chat.
        real_blender = Path.home()/".local/share/modules/blender/app/current/blender"
        if not real_blender.is_file(): raise RuntimeError("Installed Blender executable missing")
        common = [real_blender, "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1"]
        step("prepare", common+["--python", MOD/"prepare_mesh.py", "--", input_path, prepared, str(triangles)], prepared)
        record["prepared_mesh_sha256"] = sha256(prepared)
        record["preparation"] = json.loads(Path(str(prepared)+".json").read_text())
        with gpu_lease(restore_chat) as lease_fd:
            env["AI_GPU_LOCK_FD"] = str(lease_fd)
            was_active = chat_state() == "active"
            record["chat_was_active"] = was_active
            save()  # Preserve the recovery obligation before stopping anything.
            stop_chat()
            step("free_gpu", ["ai", "free"])
            monitor.thread.start()
            record["status"] = "running"; save()
            py = sys.executable
            npz = temporary/"npz"; npz.mkdir()
            def extract(input_file, label):
                code = ("import numpy as np,random,runpy;np.random.seed("+str(args.seed)+
                        ");random.seed("+str(args.seed)+");runpy.run_module('src.data.extract',run_name='__main__')")
                step(label, [py,"-c",code,"--config=configs/data/quick_inference.yaml",
                    "--require_suffix=glb,fbx","--force_override=true","--num_runs=1","--id=0",
                    "--time="+job_id,"--faces_target_count="+str(profile["inference_faces"]),
                    "--input="+str(input_file),"--output_dir="+str(npz)])
            extract(prepared, "extract_mesh")
            skeleton = temporary/"skeleton.fbx"
            step("skeleton", [py,source/"run.py","--task=configs/task/"+profile["tasks"]["skeleton"],
                 "--seed="+str(args.seed),"--input="+str(prepared),"--output="+str(skeleton),"--npz_dir="+str(npz)], skeleton)
            extract(skeleton, "extract_skeleton")
            skin = temporary/"skin.fbx"
            step("skin", [py,source/"run.py","--task=configs/task/"+profile["tasks"]["skin"],"--seed="+str(args.seed),
                 "--input="+str(skeleton),"--output="+str(skin),"--npz_dir="+str(npz),"--data_name=raw_data.npz"], skin)
            rigged = temporary/"rigged.glb"
            step("merge", [py,"-m","src.inference.merge","--require_suffix=fbx,glb","--num_runs=1","--id=0",
                 "--source="+str(skin),"--target="+str(prepared),"--output="+str(rigged)], rigged)
            prefix = output/"model"
            step("finish", [blender,"-b","--factory-startup","--disable-autoexec","--python-exit-code","1",
                 "--python",MOD/"finish.py","--",rigged,prefix,str(args.height),"--smooth-joints","--smooth-iterations",str(smooth),
                 "--facing="+args.facing], output/"model_unreal.fbx")
            lines = (logs/"finish.log").read_text(errors="replace").splitlines()
            finishes = [json.loads(line[len("RIG-FINISH "):]) for line in lines if line.startswith("RIG-FINISH ")]
            if len(finishes) != 1 or not finishes[0]["ok"]:
                raise CheckFailure("Rig finish pose/contract checks failed")
            finish_file = output/"finish.json"; finish_file.write_text(json.dumps(finishes[0], indent=2))
            record["validation_results"] = {}
            for kind, file in (("fbx", output/"model_unreal.fbx"), ("glb", output/"model_named.glb")):
                report = output/("verify_"+kind+".json")
                step("verify_"+kind, common+["--python",MOD/"verify.py","--",file,report,"--height",str(args.height),"--renders",finish_file], report)
                result = json.loads(report.read_text())
                record["validation_results"][kind] = result
                if not result["ok"]: raise CheckFailure("Re-import validation failed: " + kind)
            monitor.stop.set(); monitor.thread.join(timeout=6)
            record["peak_vram_mib"] = monitor.peak
            if not 0 < monitor.peak <= profile["max_peak_vram_mib"]:
                raise CheckFailure("GPU peak is missing or above profile budget")
            for file in (output/"model_unreal.fbx", output/"model_named.glb", output/"model_rig.blend"):
                record["outputs"][file.name] = {"sha256":sha256(file), "bytes":file.stat().st_size}
            record["status"] = "succeeded"
    except (KeyboardInterrupt, InterruptedError):
        record.update(status="interrupted", error="Job interrupted; outputs retained")
    except CheckFailure as error:
        record.update(status="check_failed", error=str(error))
    except Exception as error:
        record.update(status="failed", error=str(error))
    finally:
        monitor.stop.set()
        if monitor.thread.is_alive(): monitor.thread.join(timeout=6)
        record["peak_vram_mib"] = max(record["peak_vram_mib"], monitor.peak)
        save()
    RESULT.update(status=record["status"], record=str(record_file),
                  outputs=[str(output/name) for name in record["outputs"]], error=record.get("error"))
    if not args.json:
        print("RIG " + json.dumps(dict(status=record["status"], output=str(output), record=str(record_file))), flush=True)
    return {"succeeded":0, "check_failed":2, "interrupted":130}.get(record["status"], 1)


def main():
    def interrupted(signum, frame):
        raise InterruptedError("Signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    RESULT.clear()
    structured = "--json" in sys.argv[1:]
    with contextlib.redirect_stdout(sys.stderr) if structured else contextlib.nullcontext():
        try:
            args = parser().parse_args()
            code = execute(args)
        except SystemExit as stopped:
            if not structured: raise
            code = stopped.code if isinstance(stopped.code, int) else 1
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            print("Rig error:", error, file=sys.stderr)
            RESULT.update(error=str(error))
            code = 1
    if structured:
        result = dict(schema="local-ai.producer/1", status="failed", record=None, outputs=[])
        result.update(RESULT, exit_code=code)
        print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
