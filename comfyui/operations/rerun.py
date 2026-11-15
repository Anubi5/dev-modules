"""!
@file rerun.py
@brief ai gen rerun: run a workflow that ai gen saved, exactly, after checking it against its record.
"""
import argparse
import datetime
import json
import os
import re
import sys

import runtime as rt
from operations import model3d

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the comfyui module folder


def rerun(argv):
    """!
    @brief ai gen rerun FILE.workflow.json [--name N] [--allow-drift]: runs a workflow ai gen saved again, exactly.
           Only workflows with their record beside them and unchanged since (workflow SHA-256 in the record), made of
           node types the pinned ComfyUI's schema lists; model files and input pictures must still have the hashes
           recorded (--allow-drift runs anyway and records the difference). Input pictures are handed over again from
           where the record says; a 3D job gets the same checks as the original. @return exit code
    """
    ap = argparse.ArgumentParser(prog="ai gen rerun")
    ap.add_argument("workflow"); ap.add_argument("--name", default=""); ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--allow-drift", action="store_true", help="run even when a model file or input picture changed")
    ap.add_argument("--fast", action="store_true", help="compare cached hashes (size and time) instead of re-reading files")
    a = ap.parse_args(argv)
    p = os.path.realpath(os.path.expanduser(a.workflow))
    if not p.startswith(rt.WORK + os.sep) or not p.endswith(".workflow.json"):
        sys.exit("ai gen rerun: a .workflow.json inside ~/work (e.g. ~/work/outputs/comfy/<date>/x_00001_.workflow.json)")
    recp = p[: -len(".workflow.json")] + ".json"
    try:
        old = json.load(open(recp))
    except (OSError, ValueError):
        sys.exit(f"ai gen rerun: no record {os.path.basename(recp)} beside it - only workflows saved by ai gen can be rerun")
    wf = json.load(open(p))
    if old.get("workflow") != os.path.basename(p) or old.get("workflow_sha256") != rt.wf_sha(wf):
        sys.exit("ai gen rerun: the workflow does not match its record (changed since, or saved before records had hashes)")
    try:
        known = set(json.load(open(os.path.join(HERE, "schema.json"))))
    except (OSError, ValueError):
        known = set()
    odd = sorted({n["class_type"] for n in wf.values()} - known)
    if odd:
        sys.exit(f"ai gen rerun: node types ai gen does not use: {', '.join(odd)} - not run")
    uploads = [(n, s) for n, s in old.get("uploads", [])]
    log = lambda s: print(f"  {s}", flush=True)
    now = rt.input_hashes(wf, uploads, log, fresh=not a.fast)
    drift = sorted(k for k in set(now) | set(old.get("inputs_sha256", {})) if now.get(k) != old.get("inputs_sha256", {}).get(k))
    if drift and not a.allow_drift:
        sys.exit(f"ai gen rerun: changed or missing since the original: {', '.join(drift)} (--allow-drift runs anyway)")
    day = datetime.date.today().isoformat()
    prefix = f"{day}/{rt.slug(a.name) if a.name else 'rerun-' + rt.slug(os.path.basename(p).split('_0')[0])}"
    for node in wf.values():                       # outputs go to today's folder under the new name, nothing else changes
        if node.get("class_type", "").startswith("Save") and "filename_prefix" in node.get("inputs", {}):
            oldp = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = prefix + (oldp[oldp.rfind("_"):] if re.search(r"_(base|pose|hires|upscaled)$", oldp) else "")
    info = {k: v for k, v in old.items() if k not in ("time", "status", "error", "job", "outputs", "duration_s", "peak_gpu_gb",
                                                       "workflow", "workflow_sha256", "inputs_sha256", "glb", "blender", "renders_check")}
    info.update(rerun_of=p, drift=drift or None)
    rt.JOB["pending"] = (info, wf, prefix)
    if old.get("kind") == "3d":
        status, files, err = model3d.job_3d(wf, uploads, info, a.timeout, log)
        return model3d.report_3d(status, files, info, wf, err, prefix, old.get("seed", 0), log)
    ok, files, err = rt.generate(wf, a.timeout, log=log, uploads=uploads)
    rec = rt.write_record(files, info, wf, "succeeded" if ok else "failed", err, prefix, log)
    print("OUTPUT" if ok else f"FAILED {err}")
    for f in files:
        print(f)
    if rec:
        print(f"  record: {rec}")
    return 0 if ok else 1
