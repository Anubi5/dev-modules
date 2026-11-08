"""Finite local pipeline. Producers own GPU work; this process owns stage state."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from contracts import EXTENSIONS, atomic_json, digest, inside, read, write


SCHEMA = "local-ai.pipeline/1"


class UnfinishedStage(RuntimeError):
    pass
FLAGS = {
    "image": {"prompt", "model", "seed", "steps", "size", "negative", "style", "no_expand", "upscale"},
    "video": {"prompt", "model", "seed", "steps", "frames", "negative"},
    "mesh": {"model", "seed", "quality", "faces", "texture", "keep_background"},
    "rig": {"height", "triangles", "smooth_iterations", "facing", "seed"},
    "animation": set(),
}


def validate_plan(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("stages"), list) or not data["stages"]:
        raise ValueError("Unsupported or empty pipeline")
    seen = set()
    for stage in data["stages"]:
        if not isinstance(stage, dict): raise ValueError("Stage must be an object")
        identity, operation = stage.get("id", ""), stage.get("operation")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", identity) or identity in seen or not isinstance(operation, str) or operation not in FLAGS:
            raise ValueError("Invalid stage id or operation")
        parameters = stage.get("parameters", {})
        if not isinstance(parameters, dict) or set(parameters)-FLAGS[operation]:
            raise ValueError("Unknown stage parameter: " + identity)
        if any(not isinstance(v, (str, int, float, bool)) for v in parameters.values()):
            raise ValueError("Stage parameters must be scalar values")
        booleans = {"no_expand", "upscale", "keep_background"}
        if any(isinstance(value, bool) != (key in booleans) for key, value in parameters.items()):
            raise ValueError("Boolean stage flags must be explicit true/false values")
        timeout = stage.get("timeout_seconds", 7200)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 7200:
            raise ValueError("Stage timeout must be 1-7200 seconds")
        inputs = stage.get("inputs", [])
        if not isinstance(inputs, list): raise ValueError("Stage inputs must be a list")
        for item in inputs:
            if not isinstance(item, dict) or len(item) != 1 or set(item) not in ({"stage"}, {"manifest"}):
                raise ValueError("Input must name a prior stage or a manifest")
            if "stage" in item and (not isinstance(item["stage"], str) or item["stage"] not in seen):
                raise ValueError("Forward or missing stage reference")
            if "manifest" in item and not isinstance(item["manifest"], str):
                raise ValueError("Manifest reference must be a string")
        seen.add(identity)
    return data


def command(stage, inputs):
    operation = stage["operation"]
    paths = [(data["kind"], file) for data, files in inputs for file in files]
    if operation == "rig":
        models = [p for kind, p in paths if kind == "mesh" and p.suffix.lower() == ".glb"]
        if len(models) != 1: raise ValueError("Rig stage needs exactly one GLB mesh")
        args = ["ai", "rig", str(models[0]), "--json", "--name", stage["id"]]
    else:
        args = ["ai", "gen", "3d" if operation == "mesh" else operation]
        args += ["--json", "--name", stage["id"]]
        if operation == "mesh":
            images = [p for kind, p in paths if kind == "image"]
            if not images: raise ValueError("Mesh stage needs image inputs")
            for file in images: args += ["--image", str(file)]
    for key, value in stage.get("parameters", {}).items():
        if key == "prompt": continue
        flag = "--"+key.replace("_", "-")
        if isinstance(value, bool):
            if value: args.append(flag)
        else: args.append(flag+"="+str(value))
    if operation != "rig" and stage.get("parameters", {}).get("prompt") is not None:
        args += ["--", str(stage["parameters"]["prompt"])]
    return args


def execute(args, log, environment, timeout, started):
    with log.open("w", encoding="utf-8") as stream, log.with_suffix(".json").open("w", encoding="utf-8") as result:
        process = subprocess.Popen(args, stdout=result, stderr=stream, stdin=subprocess.DEVNULL,
                                   env=environment, start_new_session=True)
        try:
            started(process.pid)
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Producer timed out; outputs retained: " + str(log)) from error
        finally:
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
    if code: raise RuntimeError("Producer exited " + str(code) + "; see " + str(log))


def producer_outputs(log, operation, root):
    result = json.loads(log.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema") != "local-ai.producer/1" or result.get("exit_code") != 0 or result.get("status") != "succeeded":
        raise ValueError("Producer did not return a successful JSON result")
    record = result.get("record")
    if not isinstance(record, str) or not record: raise ValueError("Producer returned no record")
    path = Path(record).resolve()
    if not (path.is_relative_to(Path.home()/"work") or path.is_relative_to(root)):
        raise ValueError("Producer record is outside its work directories")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "succeeded": raise ValueError("Producer record is not succeeded")
    if operation == "rig":
        checks = data.get("validation_results", {})
        if not isinstance(checks, dict) or not checks or not all(isinstance(v, dict) and v.get("ok") is True for v in checks.values()):
            raise ValueError("Rig record does not establish numerical acceptance")
        if not isinstance(data.get("outputs"), dict): raise ValueError("Invalid rig output record")
        # A rig hand-off is FBX-only. GLB/Blender files remain in the producer
        # record; they do not share the FBX consumption units/axes declaration.
        files = [path.parent/name for name in data["outputs"] if Path(name).suffix.lower() == ".fbx"]
    else:
        if operation == "mesh" and not (isinstance(data.get("glb"), dict) and data["glb"].get("valid") is True):
            raise ValueError("Mesh record does not establish GLB validity")
        if not isinstance(data.get("outputs"), list) or any(not isinstance(p, str) for p in data["outputs"]):
            raise ValueError("Invalid generation output record")
        files = [Path(name).resolve() for name in data.get("outputs", []) if Path(name).suffix.lower() in EXTENSIONS[operation]]
    if not files: raise ValueError("Producer returned no supported output")
    for file in files:
        if not file.is_file() or not (file.resolve().is_relative_to(Path.home()/"work") or file.resolve().is_relative_to(root)):
            raise ValueError("Invalid producer output path")
    return path, files


def run(plan, workspace, resume=False):
    plan = validate_plan(plan); root = Path(workspace).resolve(); root.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    with (root/"pipeline.lock").open("a+") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another pipeline owns this workspace")
        state_file = root/"pipeline-state.json"
        if state_file.exists():
            if not resume: raise ValueError("Workspace has a run; use --resume explicitly")
            state = json.loads(state_file.read_text())
            if state.get("schema") != "local-ai.pipeline-state/1" or state.get("plan_sha256") != checksum:
                raise ValueError("Pipeline plan or state schema changed")
        else:
            state = dict(schema="local-ai.pipeline-state/1", plan_sha256=checksum, stages={})
        state["status"] = "running"; state.pop("error", None); active = None
        try:
            for stage in plan["stages"]:
                identity, operation = stage["id"], stage["operation"]
                active = state["stages"].setdefault(identity, dict(status="waiting", attempts=0))
                manifest = root/(identity+".asset.json")
                if active["status"] == "succeeded":
                    if digest(manifest) != active.get("manifest_sha256"): raise ValueError("Stage manifest changed")
                    read(manifest); continue
                if active["status"] == "running":
                    raise UnfinishedStage("Interrupted producer requires manual inspection before replay: " + identity)
                inputs, input_hashes = [], []
                for item in stage.get("inputs", []):
                    path = root/(item["stage"]+".asset.json") if "stage" in item else inside(root, item["manifest"])
                    data = read(path); inputs.append((data, [inside(path.parent, row["path"]) for row in data["files"]]))
                    input_hashes.append(dict(manifest=path.relative_to(root).as_posix(), sha256=digest(path)))
                if "inputs" in active and active["inputs"] != input_hashes: raise ValueError("Stage inputs changed")
                active["inputs"] = input_hashes
                if operation == "animation":
                    if len(inputs) != 1 or inputs[0][0]["kind"] != "rig":
                        raise ValueError("Animation stage needs one rig manifest")
                    request_file = root/(identity+".unreal-request.json")
                    if manifest.exists():
                        response = read(manifest)
                        if response["kind"] != "animation" or (response.get("source") or {}).get("request_sha256") != digest(request_file):
                            raise ValueError("Animation response does not match its hand-off request")
                    else:
                        request = dict(schema="local-ai.unreal-request/1", input_manifest=input_hashes[0]["manifest"],
                                       input_sha256=input_hashes[0]["sha256"],
                                       output_manifest=manifest.name, profile="mannequin-walk-jog")
                        atomic_json(request_file, request)
                        active["status"] = "awaiting_host"; state["status"] = "awaiting_host"
                        atomic_json(state_file, state); return state
                else:
                    args = command(stage, inputs); active["attempts"] += 1
                    output = root/"outputs"/(identity+"-"+str(active["attempts"])); output.mkdir(parents=True)
                    log = output/"producer.log"; active.update(status="running", command=args)
                    atomic_json(state_file, state)
                    def started(pid):
                        try: birth = Path("/proc/"+str(pid)+"/stat").read_text().rsplit(")", 1)[1].split()[19]
                        except (OSError, IndexError): birth = None
                        active.update(pid=pid, proc_start=birth, started=time.time()); atomic_json(state_file, state)
                    execute(args, log, dict(os.environ, RIG_WORK_DIR=str(root/"producer-work")),
                            stage.get("timeout_seconds", 7200), started)
                    record, files = producer_outputs(log, operation, root)
                    retained_record = output/"record.json"; shutil.copyfile(record, retained_record)
                    copied = []
                    for index, file in enumerate(files):
                        destination = output/(str(index)+"-"+file.name); shutil.copyfile(file, destination)
                        copied.append(("output-"+str(index), destination))
                    coordinates = None if operation in {"image", "video"} else dict(units="cm" if operation == "rig" else "m", up="+Z", forward="+X" if operation == "rig" else "-Y")
                    write(manifest, operation, copied, coordinates, numerical="passed" if operation in {"mesh", "rig"} else "not_run",
                          source=dict(record=retained_record.relative_to(root).as_posix(), sha256=digest(retained_record)))
                active.update(status="succeeded", manifest=manifest.name, manifest_sha256=digest(manifest)); atomic_json(state_file, state)
            state["status"] = "succeeded"
        except (KeyboardInterrupt, InterruptedError):
            state["status"] = "interrupted"
            if active: active["status"] = "interrupted"
            atomic_json(state_file, state); raise
        except Exception as error:
            state.update(status="failed", error=str(error))
            if active: active.setdefault("failures", []).append(dict(time=time.time(), error=str(error)))
            if active and active["status"] == "running" and not isinstance(error, UnfinishedStage): active["status"] = "failed"
            atomic_json(state_file, state); raise
        atomic_json(state_file, state)
        return state
