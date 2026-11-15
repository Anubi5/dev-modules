"""!
@file model3d.py
@brief ai gen 3d: a textured 3D model (GLB) from one or more pictures through the model's recipe, then the required
       checks (GLB header; with the Blender module: import, mesh measures, 4 renders, img-check) under one GPU lock.
"""
import datetime
import json
import os
import random
import shutil
import subprocess
import sys

import runtime as rt
from generation.graph import apply_recipe
from generation.mesh import QUALITY, glb_stats, EXIT, check_status
from generation.models import PROFILES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the comfyui module folder


def run(a):
    """!
    @brief ai gen 3d --image PHOTO: a textured 3D model (GLB) from one picture - or, for a multi-view model, from up to
           as many pictures as its recipe has image inputs (in their order, e.g. front, left, back, right) - through the
           model's recipe (official ComfyUI template, core nodes only), with the same GPU hand-off, preflight, record and saved workflow as
           every job; afterwards the GLB is measured and, when the Blender module is installed, rendered from 4 sides
           and image-checked. @return exit code
    """
    if not a.image:
        sys.exit('ai gen 3d needs --image PHOTO (one object, plain background works best; in ~/work)')
    name = a.model or PROFILES.get("default_3d", "pixal3d")
    prof = PROFILES["models"].get(name)
    if not prof or prof.get("kind") != "3d":
        sys.exit(f"--model {name}: 3D models are {', '.join(n for n, p in PROFILES['models'].items() if p.get('kind') == '3d')}")
    recipe = rt.load_recipe(prof["workflow"])
    slots = [k for k, sp in recipe["inputs"].items() if sp.get("upload")]
    if len(a.image) > len(slots):
        sys.exit(f"--model {name} takes {'one picture' if len(slots) == 1 else f'up to {len(slots)} pictures, in this order: ' + ', '.join(slots)}")
    srcs = {k: rt.check_input(f) for k, f in zip(slots, a.image)}
    src = srcs[slots[0]]
    seed = a.seed if a.seed is not None else random.randint(0, 2**31 - 1000)
    day = datetime.date.today().isoformat()
    prefix = f"{day}/{rt.slug(a.name) if a.name else rt.slug(os.path.splitext(os.path.basename(src))[0])}"
    vals = dict(QUALITY[a.quality], **{k: v for k, v in (("faces", a.faces), ("texture", a.texture)) if v is not None})
    vals.update(prof.get("set", {}), seed=seed, prefix=prefix, remove_background=not a.keep_background, **{k: "pending" for k in srcs})
    wf = apply_recipe(recipe, vals)
    log = lambda s: print(f"  {s}", flush=True)
    log(f"{name}: {a.quality} quality ({vals['faces']} faces, upsample {vals['resolution']}, texture {vals['texture']}), seed {seed}")
    uploads = [(recipe["inputs"][k]["set"][0][0], f) for k, f in srcs.items()]
    info = {"kind": "3d", "model": name, "model_file": prof["file"], "image": a.image[0], "seed": seed, "quality": a.quality,
            **({"views": dict(zip(slots, a.image))} if len(slots) > 1 else {}),
            **{k: vals[k] for k in ("faces", "resolution", "texture", "remove_background")}, "recipe": prof["workflow"],
            "uploads": [list(u) for u in uploads]}
    rt.JOB["pending"] = (info, wf, prefix)          # a stop before the end still gets a record (entry)
    status, files, err = job_3d(wf, uploads, info, a.timeout or 2400, log)
    return report_3d(status, files, info, wf, err, prefix, seed, log)


def job_3d(wf, uploads, info, timeout, log=print):
    """!
    @brief A 3D job as one unit: the GPU lock is held from generation through the Blender check, and the job file gets
           its final state only from the checks' verdict. @return (status, output files, error text)
    """
    files, extra = [], []
    try:
        with rt.gpu_lease() as lock:
            try:
                ok, files, err = rt.generate(wf, timeout, log=log, uploads=uploads, lease=lock, finish=False)
            except Exception as e:                     # whatever breaks, the job ends with a state and a record
                ok, err = False, f"generation stopped: {type(e).__name__}: {str(e)[:300]}"
            glb = next((f for f in files if f.endswith(".glb")), None)
            if ok and not glb:
                err = "ComfyUI finished but saved no GLB"
            status = "failed"
            if ok and glb:
                try:
                    status, extra = check_3d(glb, info, log, lock)
                except Exception as e:
                    status, info["check_error"] = "check_failed", f"{type(e).__name__}: {str(e)[:300]}"
                    info.setdefault("glb", {"valid": False, "error": "not read"})
    except rt.GpuBusy as e:
        rt.job_state("failed", error=str(e))
        return "failed", [], str(e)
    files = files + extra
    rt.job_state(status, outputs=files, error=None if status == "succeeded" else err or status)
    return status, files, err


def handover_ok(blender):
    """! @brief Whether the blender command accepts the GPU lock handed over by its caller (AI_GPU_LOCK_FD). """
    try:
        with open(blender, "rb") as f:
            return b"AI_GPU_LOCK_FD" in f.read(1 << 16)
    except OSError:
        return False


def check_3d(glb, info, log=print, lock=None):
    """!
    @brief Required checks of a 3D result: GLB header, then (Blender module) import, mesh measures, 4 renders and
           img-check, with the chat-model server paused (the blender command takes the GPU lock itself).
    @return (status, extra output files)
    """
    info["glb"] = glb_stats(glb)
    blender, views, rc, mesh, chk_rc = shutil.which("blender"), [], None, None, None
    if blender and lock and not handover_ok(blender):
        info["blender"] = {"ok": False, "error": "the Blender module is too old to take over the GPU lock this job holds "
                                                 "(it would wait for it until the time limit): ai mod update blender"}
        return check_status(info["glb"], True, None, info["blender"], 0, None), []
    if info["glb"].get("valid") and info["glb"].get("triangles") and blender:
        log("checking the model in Blender (import, mesh, 4 renders)")
        try:
            paused = rt.chat_pause(log)
        except RuntimeError as e:
            info["blender"] = {"ok": False, "error": str(e)}
            return "check_failed", []
        fd = lock.fileno() if lock else None          # the held lock is handed to blender, which then does not wait for itself
        env = dict(os.environ, AI_GPU_LOCK_FD=str(fd)) if fd is not None else None
        try:
            r = subprocess.run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                                os.path.join(HERE, "mesh_check.py"), "--",
                                glb, os.path.splitext(glb)[0] + "_view"], capture_output=True, text=True, timeout=900,
                               pass_fds=(fd,) if fd is not None else (), env=env)
            rc = r.returncode
            line = next((l for l in r.stdout.splitlines() if l.startswith("MESH-CHECK ")), None)
            mesh = json.loads(line[11:]) if line else {"ok": False, "error": (r.stderr or r.stdout)[-300:]}
        except (subprocess.TimeoutExpired, ValueError) as e:
            mesh = {"ok": False, "error": str(e)[:300]}
        finally:
            rt.chat_resume(paused, log)
        views = [v for v in mesh.get("renders", []) if os.path.exists(v)]
        if views:
            chk = subprocess.run([os.path.join(HERE, "bin", "img-check"), "--min-coverage", "0.02", *views],
                                 capture_output=True, text=True)
            chk_rc, info["renders_check"] = chk.returncode, chk.stdout.strip()[-600:]
        info["blender"] = mesh
    elif not blender:
        info["blender"] = "not installed (ai mod add blender for renders and mesh checks)"
    return check_status(info["glb"], bool(blender), rc, mesh, len(views), chk_rc), views


def report_3d(status, files, info, wf, err, prefix, seed, log=print):
    """! @brief Record and summary for a 3D job; the exit code says whether the required checks passed. """
    why = {"failed": err or "no valid model" + (f" ({info['glb']['error']})" if (info.get("glb") or {}).get("error") else ""), "check_failed": "the model was made but a required check failed (see the record)",
           "check_skipped": "the model was made but not checked (Blender module not installed)"}.get(status)
    rec = rt.write_record(files, info, wf, status, why, prefix, log)
    print({"succeeded": f"OUTPUT (seed {seed})", "check_skipped": f"OUTPUT, NOT CHECKED (seed {seed})",
           "check_failed": f"OUTPUT, CHECK FAILED (seed {seed})"}.get(status, f"FAILED {why}"))
    for f in files:
        print(f)
    if status != "failed":
        s = info.get("glb") or {}
        b = info.get("blender") if isinstance(info.get("blender"), dict) else {}
        if s.get("valid"):
            print(f"  model: {s.get('triangles')} triangles, {s.get('textures')} textures, {s.get('size_mb')} MB"
                  + (f"; Blender: {b.get('loose_parts')} parts, {b.get('non_manifold_edges')} open edges" if b.get("ok") else ""))
        if status == "check_failed":
            print(f"  check failed: {info.get('check_error') or s.get('error') or ''} Blender {'ok' if b.get('ok') else b.get('error', 'failed')};"
                  f" renders: {info.get('renders_check') or 'none'}")
    if rec:
        print(f"  record: {rec}")
    return EXIT[status]
