#!/usr/bin/env python3
"""!
@file gen.py
@brief "ai gen": the command line and router. It parses, applies the limits every job shares, hands the job to its
       operation (operations/image, edit, pose, video, model3d, rerun) and runs it through the shared GPU hand-off
       (runtime.py): one GPU job at a time, chat model out, ComfyUI in, and back.

Usage:
  gen.py image "prompt" [--name N] [--negative T] [--seed S] [--steps N] [--size WxH] [--upscale] [--hires]
               [--pose IMG [--pose-strength F]] [--face IMG [--face-weight F]] [--fix-faces] [--fix-hands]
  gen.py video "prompt" [--model M] [--name N] [--negative T] [--seed S] [--steps N] [--frames N]
Human output lists produced paths; --json emits one producer result with progress on stderr.
Exit code 0 on success. Parser-derived help never inspects models or starts services.
"""
import argparse
import contextlib
import datetime
import json
import os
import random
import re
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime as rt  # noqa: E402
from runtime import JOB, FINAL, PROFILES, QUALITY  # noqa: E402,F401
from operations import edit, image, model3d, pose, rerun, video  # noqa: E402

OPERATIONS = {"image": image, "edit": edit, "pose": pose, "video": video}   # 3d and rerun run their own job


GENERATION_KINDS = ("image", "video", "edit", "pose", "3d")
ASSET_COMMANDS = ("list", "show", "check", "note", "verify", "names")
# The other "ai gen" commands, dispatched in main() and listed in help (their own --help has the details).
OTHER_COMMANDS = {
    "rerun": "run a saved workflow again, exactly",
    "jobs": "recent job records (read-only)",
    "list": "installed model files and LoRAs",
    "show": "what is known about one model file",
    "check": "check a model/checkpoint/LoRA combination before a job",
    "note": "record notes about a model file",
    "verify": "check model files against their declared hashes",
    "names": "plain names for completion",
    "schema": "save the pinned ComfyUI node definitions (starts ComfyUI)",
    "models": "readiness of every configured profile; no GPU job",
}


def generation_parser():
    """Build the same parser for execution and side-effect-free help."""
    ap = argparse.ArgumentParser(prog="ai gen", description="Generate images, video or 3D with existing model files; coordinates the shared GPU.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="other commands:\n" + "\n".join(f"  {k:<8} {v}" for k, v in OTHER_COMMANDS.items()))
    ap.add_argument("kind", choices=list(GENERATION_KINDS),
                    help="edit = change an image by instruction; pose = a ControlNet pose (photo + skeleton)")
    ap.add_argument("--json", action="store_true", help="emit one producer JSON result; progress goes to stderr")
    ap.add_argument("--check", action="store_true",
                    help="validate only (model, family, features, inputs, files, needed modules); no GPU, no service, no outputs or records")
    ap.add_argument("prompt", nargs="?", default="", help="what to make; for edit: what to change; for pose: the pose")
    ap.add_argument("--from", dest="src", metavar="PHOTO", help="pose: make the skeleton from this photo (in ~/work)")
    ap.add_argument("--pose-skeleton", action="store_true", help="image: the --pose picture is already a skeleton")
    ap.add_argument("--image", metavar="IMAGE", action="append", default=[],
                    help="edit: the image to change (in ~/work); repeat for extra references (up to 4)")
    ap.add_argument("--no-expand", action="store_true", help="FLUX 3: use the prompt as typed (no caption expansion)")
    ap.add_argument("--name", default="", help="short name for the output files (default: from the prompt)")
    ap.add_argument("--model", metavar="NAME", help=f"image/3D profile, video package or compatible model file; default {PROFILES['default']}; use ai gen list for current names")
    ap.add_argument("--style", choices=list(PROFILES["styles"]), default=None, help="image: photo or art (default: per model)")
    ap.add_argument("--negative", default=None, help='what to avoid (default: per model and style; "" = nothing)')
    ap.add_argument("--count", type=int, default=1, help="image: 1-8 sequential images; seeds increment from --seed (or one random starting seed)")
    ap.add_argument("--seed", type=int, default=None, help="integer random seed; omitted: choose a fresh random seed")
    ap.add_argument("--steps", type=int, default=None, help="image/video/edit/pose: sampling steps, 1-60; default per selected model")
    ap.add_argument("--size", default=None, help="image size WxH (multiples of 64; default 1024x1024, pose 896x1152)")
    ap.add_argument("--upscale", action="store_true", help="image: also save a fast 2x copy (4x-UltraSharp)")
    ap.add_argument("--hires", action="store_true", help="image: also save a detailed 2x copy (tiles redrawn, slower)")
    ap.add_argument("--pose", metavar="IMAGE", help="image: copy the body pose from this picture (in ~/work)")
    ap.add_argument("--pose-strength", type=float, default=None, help="how strictly to follow the pose, 0.3-1.0 (default per model)")
    ap.add_argument("--face", metavar="IMAGE", help="image: keep this face (in ~/work; one clear face)")
    ap.add_argument("--face-weight", type=float, default=0.65, help="how strongly to keep the face, 0.3-1.2 (high values distort it)")
    ap.add_argument("--checkpoint", metavar="NAME", help="image: any SDXL checkpoint file by (part of) its name")
    ap.add_argument("--cfg", type=float, default=None, help="image: guidance scale (default per model; fixed for zimage, flux3, fibo2)")
    ap.add_argument("--force-lora", action="store_true", help="image: run a LoRA even when its layers do not match the model (recorded)")
    ap.add_argument("--lora", metavar="NAME[:STRENGTH]", action="append", default=[],
                    help="image/edit: a LoRA by (part of) its file name, strength -2..2 (default 1); repeat for up to 4")
    ap.add_argument("--fix-faces", action="store_true", help="image: redraw faces in more detail")
    ap.add_argument("--fix-hands", action="store_true", help="image: redraw hands in more detail")
    ap.add_argument("--frames", type=int, default=33, help="video length, 4n+1 (33 ~ 1.4 s, 121 ~ 5 s)")
    ap.add_argument("--timeout", type=int, default=None, help="seconds (default 900 image, 3600 video)")
    ap.add_argument("--quality", choices=list(QUALITY), default="normal", help="3d: normal (fast, more GPU headroom) or high")
    ap.add_argument("--faces", type=int, default=None, help="3d: triangles in the final model (default per --quality)")
    ap.add_argument("--texture", type=int, default=None, help="3d: texture size 1024 | 2048 | 4096 (default per --quality)")
    ap.add_argument("--keep-background", action="store_true", help="3d: the picture is already cut out (no background removal)")
    return ap


def image_batch(args):
    """Run bounded image jobs sequentially, retaining each seed and stopping on failure."""
    first = args.seed if args.seed is not None else random.randint(1, 2**31)
    if not 0 <= first <= 2**64 - args.count:
        sys.exit("--seed plus --count exceeds the sampler seed range")
    results = []
    for index in range(args.count):
        item = argparse.Namespace(**vars(args))
        item.count, item.seed = 1, first + index
        item.name = f"{args.name or ' '.join(args.prompt.split()[:6])}-{index + 1:02d}"
        JOB.pop("result", None)
        code = main(item)
        results.append(dict(JOB.get("result", {}), seed=item.seed))
        JOB["result"] = dict(status="failed" if code else "succeeded", record=None,
            outputs=[path for result in results for path in result.get("outputs", [])],
            records=[result.get("record") for result in results], results=results,
            error=results[-1].get("error"), requested=args.count, completed=sum(r.get("status") == "succeeded" for r in results))
        if code: return code
    return 0


def entry(fn=None):
    """Keep human output by default; producers expose one JSON result on stdout."""
    structured = sys.argv[1:2] and sys.argv[1] in (*GENERATION_KINDS, "rerun") and "--json" in sys.argv[2:]
    if not structured:
        return _entry(fn)
    arguments = sys.argv[:]
    sys.argv.remove("--json")
    JOB.pop("result", None)
    error = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                code = _entry(fn)
            except SystemExit as stopped:
                code = stopped.code if isinstance(stopped.code, int) else 1
                error = None if isinstance(stopped.code, int) else str(stopped.code)
                if error: print(error, file=sys.stderr)
            except Exception as failed:
                code, error = 1, str(failed)
                print(error, file=sys.stderr)
        result = dict(schema="local-ai.producer/1", status="failed", record=None, outputs=[], error=error)
        result.update(JOB.get("result", {}), exit_code=code)
        print(json.dumps(result), flush=True)
        return code
    finally:
        sys.argv[:] = arguments


def _entry(fn=None):
    """!
    @brief Runs a command so that a stop at any point - Ctrl+C, the terminal closing (SIGHUP), a kill (SIGTERM) -
           leaves the job file "interrupted" instead of mid-way (SIGKILL cannot be caught: job_view shows those as
           abandoned). @return exit code
    """
    def stop(sig, frame):
        raise KeyboardInterrupt(signal.Signals(sig).name)
    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        return (fn or main)()
    except KeyboardInterrupt as e:
        j = JOB.get("ledger")
        if j and j.get("state") not in FINAL:
            rt.job_state("interrupted", error=f"stopped ({e or 'Ctrl+C'}) while {j.get('state')}")
        if JOB.get("chat_paused"):               # stopped between pausing the chat server and resuming it
            rt.chat_resume(True, lambda t: print(f"  {t}", file=sys.stderr))
        if JOB.get("pending"):                   # the record every outcome gets (a second Ctrl+C skips it)
            info, wf, prefix = JOB["pending"]
            try:
                rec = rt.write_record([], info, wf, "interrupted", (JOB.get("ledger") or {}).get("error") or "stopped", prefix,
                                   lambda t: None)
                print(f"  record: {rec}", file=sys.stderr)
            except (OSError, ValueError):
                pass
        print("stopped", file=sys.stderr)
        return 130
    finally:
        for s, h in old.items():
            signal.signal(s, h)



def main(a=None):
    """!
    @brief Command-line entry point. @return 0 on success, 1 on failure.
    """
    if sys.argv[2:3] == ["help"]:
        sys.argv[2] = "--help"
    if sys.argv[1:2] == ["rerun"]:
        return rerun.rerun(sys.argv[2:])
    if sys.argv[1:2] == ["jobs"]:
        return rt.jobs_cmd(sys.argv[2:])
    if sys.argv[1:2] and sys.argv[1] in ASSET_COMMANDS:
        return rt.asset_cmd(sys.argv[1:])
    ap = generation_parser()
    if sys.argv[1:2] == ["schema"]:
        argparse.ArgumentParser(prog="ai gen schema", description="Query ComfyUI node definitions and write the schema in the output directory.").parse_args(sys.argv[2:])
        return rt.dump_schema(os.path.join(rt.OUTPUT, "schema.json"))
    if sys.argv[1:2] == ["models"]:
        argparse.ArgumentParser(prog="ai gen models", description="Inspect readiness of configured generation profiles; no GPU job.").parse_args(sys.argv[2:])
        return rt.models_report()
    if not sys.argv[1:] or sys.argv[1] in ("help", "-h", "--help"):
        ap.print_help(); return 0
    # Optional positional prompts may follow options and '--', including literal
    # prompts beginning with '-'. Use argparse's supported intermixed parser.
    a = a or ap.parse_intermixed_args()
    if not 1 <= a.count <= 8: ap.error("--count must be 1-8")
    if a.count != 1:
        if a.kind != "image": ap.error("--count is for image generation")
        return image_batch(a)
    if a.seed is not None and not 0 <= a.seed < 2**64:
        ap.error("--seed must be 0-18446744073709551615")
    if a.kind == "3d":
        return model3d.run(a)
    # limits for a 12 GB GPU (and against a runaway agent): larger jobs fail or block the GPU for hours
    if a.steps is not None and not 1 <= a.steps <= rt.MAX_STEPS:
        sys.exit(f"--steps must be 1-{rt.MAX_STEPS}")
    if a.timeout is not None and not 60 <= a.timeout <= rt.MAX_TIMEOUT:
        sys.exit(f"--timeout must be 60-{rt.MAX_TIMEOUT} s")
    operation = OPERATIONS[a.kind]
    if hasattr(operation, "validate"):
        operation.validate(a)
    if not a.prompt.strip() and not (a.kind == "pose" and a.src):
        sys.exit(f'ai gen {a.kind} needs a prompt, e.g. ai gen {a.kind} "..."' + (" (or --from PHOTO)" if a.kind == "pose" else ""))
    if a.src and a.kind != "pose":
        sys.exit("--from is for ai gen pose (use --pose to copy a pose into an image)")
    a.size = a.size or ("896x1152" if a.kind == "pose" else "1024x1024")
    if not a.check:
        os.makedirs(rt.OUTPUT, exist_ok=True)
        if shutil.disk_usage(rt.OUTPUT).free < rt.MIN_FREE:
            sys.exit(f"less than {rt.MIN_FREE >> 30} GB free in {rt.OUTPUT}")
    seed = a.seed if a.seed is not None else random.randint(1, 2**31)
    prefix = f"{datetime.date.today():%Y-%m-%d}/{rt.slug(a.name or ' '.join(a.prompt.split()[:6]) or 'pose')}"
    wf, uploads, info, limit, prof = operation.prepare(a, seed, prefix)
    if a.check:
        return rt.checked(a, info, wf, prof, seed, getattr(a, "needs", None))
    log = lambda s: print(f"  {s}", flush=True)
    t_start = time.time()
    info["uploads"] = [[n, s] for n, s in uploads]
    JOB["pending"] = (info, wf, prefix)
    ok, files, err = rt.generate(wf, limit, log=log, uploads=uploads)
    if not ok and re.search(r"read_file_slice|HostBuffer|deserializ|Unable to load|out of memory", err or "", re.I):
        # model file read from D: failed (seen right after the chat model unloaded, and mid-job after a pose preview
        # was already saved - so earlier files do not mean the job got past loading): retry the whole job once
        log(f"model loading failed ({err[:80]}) - retrying once")
        ok, files, err = rt.generate(wf, limit, log=log, uploads=uploads)
    if err:
        print(f"  error: {err}")
    rec = rt.write_record(files, info, wf, "succeeded" if ok else "failed", err, prefix)
    JOB["result"] = dict(JOB.get("result", {}), **rt.result_fields(a, info, wf, prof, seed))
    if ok and a.kind != "video" and info.get("loras"):       # ComfyUI's verdict: did it apply every LoRA layer?
        dropped = rt.lora_keys_dropped(t_start)
        if dropped:
            print(f"  warning: ComfyUI could not apply {dropped} LoRA layer(s) in this job (ai logs comfyui) - "
                  "not recorded as tested")
        elif dropped == 0:
            mp = rt.model_path(prof)
            for d in rt.LORA_DIRS:
                for c in info["loras"]:
                    if mp and os.path.exists(os.path.join(d, c["file"])):
                        rt.assets.record_tested(os.path.join(d, c["file"]), mp)
        if rec:
            r = json.load(open(rec))
            r["lora_layers_not_applied"] = dropped
            json.dump(r, open(rec, "w"), indent=1)
    print(f"OUTPUT (seed {seed})" if ok else "FAILED")
    for f in files:
        print(f)
    if rec:
        print(f"  record: {rec}")
        print(f"  workflow: {os.path.splitext(rec)[0]}.workflow.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(entry())
