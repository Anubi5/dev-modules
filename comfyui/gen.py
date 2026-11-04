#!/usr/bin/env python3
"""!
@file gen.py
@brief GPU hand-off for image/video generation ("ai gen"): one GPU job at a time, chat model out, ComfyUI in, and back.

Steps: take the GPU lock (waits if another ai gen runs) -> make sure ComfyUI runs and wait while it runs someone
else's job -> unload the chat models and confirm none is loaded (stops if one is) -> queue a fixed, reviewed
workflow (only prompt, seed, size, length, name, strengths and reference images are filled in; limits apply) -> wait with a timeout
(cancels only this job) -> free ComfyUI's GPU memory unless another job is active -> release the lock.
Known limit: a chat request made during a job reloads the chat model; finish or avoid chatting meanwhile. The chat model loads again by
itself on the next request. Outputs land in COMFY_OUTPUT/<YYYY-MM-DD>/<name>_NNNNN.<ext>; the name is reduced to
[a-z0-9-], so it can never point outside that folder.

Usage:
  gen.py image "prompt" [--name N] [--negative T] [--seed S] [--steps N] [--size WxH] [--upscale] [--hires]
               [--pose IMG [--pose-strength F]] [--face IMG [--face-weight F]] [--fix-faces] [--fix-hands]
  gen.py video "prompt" [--name N] [--negative T] [--seed S] [--steps N] [--frames N]
Prints the output file paths (one per line, after "OUTPUT"); exit code 0 on success.
"""
import argparse
import datetime
import fcntl
import json
import os
import random
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assets  # noqa: E402
import threading
import time
import urllib.error
import urllib.request
import uuid

SETUP = "/opt/ai/setup"
NVSMI = "/usr/lib/wsl/lib/nvidia-smi"
LOCK = "/tmp/ai-gpu.lock"
MAX_SIDE, MAX_FRAMES, MAX_STEPS, MAX_TIMEOUT = 1536, 121, 60, 7200   # per-job limits (12 GB GPU)
JOB = {}                                                               # prompt id of the job in progress
MIN_FREE = 2 << 30                                                     # bytes free needed in the output folder


def env(key, default):
    """!
    @brief Reads one KEY="value" setting from sandbox.env.
    @param key      Setting name.
    @param default  Value if the file or key is missing.
    @return The value as a string.
    """
    try:
        for line in open(f"{SETUP}/sandbox.env"):
            if line.startswith(key + "="):
                return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    except OSError:
        pass
    return default


MODELS = os.environ.get("MODELS_MOUNT") or env("MODELS_MOUNT", "/models")   # read-only model share (D:)
assets.ROOTS[0] = os.path.join(MODELS, "comfy")
URL = "http://127.0.0.1:8188"   # this module's port (comfyui.service)
LLM = f"http://127.0.0.1:{env('LLM_PORT', '8080')}"
OUTPUT = os.path.expanduser("~/work/outputs/comfy")


def api(path, body=None, timeout=30, base=None):
    """!
    @brief Calls a ComfyUI (or model server) API path.
    @param path     API path, e.g. "/prompt".
    @param body     Dict to POST as JSON, or None for GET.
    @param timeout  Seconds to wait.
    @param base     Server address (default: ComfyUI).
    @return Parsed JSON answer (or {} if the answer is not JSON).
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request((base or URL) + path, data, {"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    try:
        return json.loads(raw)
    except ValueError:
        return {}


IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
WORK = os.path.realpath(os.path.expanduser("~/work"))


def check_input(path):
    """!
    @brief A reference image must be a picture inside ~/work (the agent workspace), at most 30 MB.
    @param path  Path given on the command line.  @return The real path, or exits with a message.
    """
    real = os.path.realpath(os.path.expanduser(path))
    if not (real.startswith(WORK + os.sep) and os.path.isfile(real)):
        sys.exit(f"{path}: must be an existing file inside ~/work (put it in ~/work/inputs/)")
    if os.path.splitext(real)[1].lower() not in IMAGE_TYPES or os.path.getsize(real) > 30 << 20:
        sys.exit(f"{path}: must be a .png/.jpg/.webp image of at most 30 MB")
    return real


def upload(src):
    """!
    @brief Hands a reference image to ComfyUI as a temporary input (POST /upload/image, type temp; cleared on restart).
    @param src  Checked file path.  @return The image name for LoadImage ("gen/<id>.png [temp]").
    """
    ext = os.path.splitext(src)[1].lower()
    name, bnd = f"{uuid.uuid4().hex}{ext}", uuid.uuid4().hex
    parts = [f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in (("type", "temp"), ("subfolder", "gen"), ("overwrite", "true"))]
    parts.append(f'--{bnd}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                 f'Content-Type: {IMAGE_TYPES[ext]}\r\n\r\n'.encode() + open(src, "rb").read() + b"\r\n")
    req = urllib.request.Request(URL + "/upload/image", b"".join(parts) + f"--{bnd}--\r\n".encode(),
                                 {"Content-Type": f"multipart/form-data; boundary={bnd}"})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return f"{r.get('subfolder') + '/' if r.get('subfolder') else ''}{r['name']} [temp]"


def slug(name):
    """!
    @brief Makes a safe file name part: lower case, [a-z0-9-] only, at most 40 characters.
    @param name  Free text.
    @return The safe name ("untitled" if nothing is left).
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "untitled"


PROFILES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")))
NEGATIVE = PROFILES["styles"]["photo"]["negative"]            # default style; kept as a name for the tests


def profile(name=None):
    """! @brief Settings for a model from profiles.json (default model when name is None). @return dict incl. "name" """
    name = name or PROFILES["default"]
    if name not in PROFILES["models"]:
        sys.exit(f"--model {name}: unknown. Models: {', '.join(PROFILES['models'])}")
    return dict(PROFILES["models"][name], name=name)


def negative_for(prof, style):
    """! @brief Default negative prompt for a model and style (the model's own, else the style's). @return text """
    return prof.get("negative", {}).get(style, PROFILES["styles"][style]["negative"])


def is_sdxl(path):
    """! @brief Reads a .safetensors header and tells whether it is an SDXL checkpoint (has SDXL's 2nd text encoder). """
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            head = f.read(min(n, 50 << 20)).decode("utf-8", "ignore")
        return "conditioner.embedders.1.model" in head
    except OSError:
        return False


def find_checkpoint(name, log=print):
    """! @brief --checkpoint NAME: the same as --model NAME, limited to the checkpoints folder (kept for old commands). """
    return resolve_model(name, log, folders=("checkpoints",))


def arch_base(arch):
    """! @brief The shipped profile an architecture's other files start from (default_sdxl for SDXL, else the first). """
    if arch == "sdxl":
        return profile(PROFILES.get("default_sdxl", "realvis"))
    name = next((n for n, p in PROFILES["models"].items() if p["arch"] == arch), None)
    if not name:
        sys.exit(f"architecture {arch!r} is not wired into ai gen. Known: {', '.join(sorted(archs()))}")
    return profile(name)


def archs():
    """! @brief Image architectures ai gen has workflows for (3D models run through ai gen 3d). """
    return {p["arch"] for p in PROFILES["models"].values() if p.get("kind") != "3d"}


RECIPES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")


def load_recipe(name):
    """! @brief A recipe: a tested ComfyUI workflow (API format) with declared inputs. @return dict """
    return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name)))


def apply_recipe(recipe, values):
    """!
    @brief Fills a recipe's declared inputs and nothing else. Each input names the node inputs it sets
           ([node, input] or [node, input, offset] for seeds that differ per stage); values outside an input's
           declared range or choices are refused. @param values  {input name: value}  @return workflow (a copy)
    """
    wf = json.loads(json.dumps(recipe["workflow"]))
    for name, spec in recipe["inputs"].items():
        v = values.get(name, spec.get("default"))
        if v is None:
            continue
        if "choices" in spec and v not in spec["choices"]:
            sys.exit(f"{name} {v}: choose one of {', '.join(map(str, spec['choices']))}")
        if ("min" in spec and v < spec["min"]) or ("max" in spec and v > spec["max"]):
            sys.exit(f"{name} {v}: must be {spec.get('min')}-{spec.get('max')}")
        for node, key, *off in spec["set"]:
            wf[node]["inputs"][key] = v + off[0] if off else v
    return wf


def resolve_model(name, log=print, folders=("checkpoints", "diffusion_models")):
    """!
    @brief --model NAME: one rule for every model. NAME is a profile (profiles.json: the shipped, tested settings) or
           any model file in checkpoints/ or diffusion_models/ (part of its name). A file's architecture comes from
           its note (ai gen note FILE --arch ...) or, for SDXL checkpoints, from its header; its settings start from
           that architecture's shipped profile. The file's note then replaces any of them (sampler, scheduler, steps,
           cfg, negative, text_encoder, vae, ...); typed options win over both later.
    @return profile dict with "name", "file", "arch" and the settings
    """
    if PROFILES["models"].get(name, {}).get("kind") == "3d":
        sys.exit(f"{name} makes 3D models: ai gen 3d --image PHOTO --model {name}")
    if name in PROFILES["models"] and "checkpoints" in folders and "diffusion_models" in folders:
        prof = profile(name)
        path = model_path(prof)
        return apply_note(prof, path, log) if path else prof
    fs = [f for d in folders for f in assets.files(d) if f["name"].endswith((".safetensors", ".gguf"))]
    exact = [f for f in fs if f["name"].lower() in (name.lower(), name.lower() + ".safetensors")]
    hit = exact or [f for f in fs if name.lower() in f["name"].lower()]
    if len(hit) != 1:
        sys.exit(f"--model {name}: {'ambiguous: ' + ', '.join(f['name'] for f in hit) if hit else 'not found'}. "
                 f"Profiles: {', '.join(PROFILES['models'])}; files: ai gen list")
    f = hit[0]
    known = [k for k, v in PROFILES["models"].items() if v["file"] == f["name"]]
    if known:
        return apply_note(profile(known[0]), f["path"], log)
    note = assets.note_for(_announce_hash(f["path"], log))
    cls = detect_classes([f["path"]]).get(f["path"])
    arch = note.get("arch") or class_arch().get(cls)
    if not arch:
        sys.exit(f"{f['name']}: ComfyUI recognises it as {cls or 'an unknown kind'}, which no ai gen workflow handles"
                 + (" yet" if cls else "") + f". If it is one of {', '.join(sorted(archs()))}: ai gen note {f['name']} --arch ...")
    want = "checkpoints" if arch == "sdxl" else "diffusion_models"
    if f["folder"] != want:
        sys.exit(f"{f['name']} is in {f['folder']}/, but {arch} models are loaded from {want}/ - move it there")
    stem = os.path.splitext(os.path.basename(f["name"]))[0]
    base = {k: v for k, v in arch_base(arch).items() if k not in ("about", "licence", "source", "optional", "negative")}
    return apply_note(dict(base, file=f["name"], name=stem), f["path"], log)


DETECT_CACHE = os.path.expanduser("~/.cache/ai-gen/detect.json")     # path -> [size, mtime, class]; disposable
COMFY_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "ComfyUI", ".venv", "bin", "python")


def detect_classes(paths):
    """!
    @brief ComfyUI's own model detection for these files (detect.py, CPU, headers only), cached by path, size and time.
    @return {path: ComfyUI model class name or None}
    """
    cache, out, todo = assets._load(DETECT_CACHE), {}, []
    for p in paths:
        st = os.stat(p)
        c = cache.get(p)
        if c and c[0] == st.st_size and c[1] == int(st.st_mtime):
            out[p] = c[2]
        else:
            todo.append(p)
    if todo:
        try:
            r = subprocess.run([COMFY_PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect.py"), *todo],
                               capture_output=True, text=True, timeout=600)
            found = json.loads(r.stdout.strip().splitlines()[-1])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            found = {}
        for p in todo:
            st = os.stat(p)
            out[p] = found.get(p)
            if p in found:
                cache[p] = [st.st_size, int(st.st_mtime), found[p]]
        assets._save(DETECT_CACHE, cache)
    return out


def class_arch():
    """!
    @brief Which ai gen workflow a ComfyUI model class runs with, learned from the shipped profiles: the class ComfyUI
           detects for each profile's file. A class seen with two workflows is left out (ambiguous). @return {class: arch}
    """
    paths = {model_path(p): p["arch"] for p in PROFILES["models"].values()}
    paths.pop(None, None)
    seen = {}
    for path, cls in detect_classes(list(paths)).items():
        if cls:
            seen.setdefault(cls, set()).add(paths[path])
    return {c: next(iter(a)) for c, a in seen.items() if len(a) == 1}


def _announce_hash(path, log):
    if path not in assets._load(assets.CACHE) and os.path.getsize(path) > 1 << 30:
        log(f"reading {os.path.basename(path)} once to recognise it by content (about a minute; cached afterwards)")
    return path


def apply_note(prof, path, log=print):
    """! @brief The file's note replaces the profile's settings it names (said in the output). @return the profile """
    note = assets.note_for(_announce_hash(path, log))
    used = {k: note[k] for k in assets.PROFILE_KEYS if k in note and k != "arch"}
    if "negative" in note:
        used["negative"] = {"photo": note["negative"], "art": note["negative"]}
    if used:
        log(f"{os.path.basename(path)}: " + ", ".join(f"{k} {note[k]!r}" for k in note if k in used) + " (from its note)")
    return dict(prof, **used)


LORA_DIRS = [os.path.join(MODELS, "comfy", "loras"),
             os.path.expanduser("~/.local/share/modules/comfyui/models/loras")]
MAX_LORAS = 4


def find_loras(specs):
    """!
    @brief --lora NAME[:STRENGTH]: LoRA files by (part of) their name in D:\\AI\\models\\comfy\\loras, strength -2..2
           (default 1.0). @param specs  The --lora values.  @return [(file name, strength)], or exits with what is there.
    """
    if len(specs) > MAX_LORAS:
        sys.exit(f"--lora: at most {MAX_LORAS}")
    found = sorted({os.path.relpath(os.path.join(r, f), d) for d in LORA_DIRS if os.path.isdir(d)
                    for r, _, fs in os.walk(d) for f in fs if f.endswith((".safetensors", ".ckpt", ".pt"))})
    out = []
    for spec in specs:
        name, _, st = spec.rpartition(":") if re.search(r":-?[\d.]+$", spec) else (spec, "", "")
        try:
            strength = float(st) if st else None          # None: the LoRA's note, else 1.0 (resolve_loras)
        except ValueError:
            sys.exit(f"--lora {spec}: strength must be a number, e.g. {name}:0.8")
        if strength is not None and not -2.0 <= strength <= 2.0:
            sys.exit(f"--lora {spec}: strength must be -2 to 2")
        exact = [f for f in found if os.path.basename(f).lower() in (name.lower(), name.lower() + ".safetensors")]
        hit = exact or [f for f in found if name.lower() in f.lower()]
        if len(hit) != 1:
            sys.exit(f"--lora {name}: {'ambiguous: ' + ', '.join(hit) if hit else 'not found'}. Available: "
                     f"{', '.join(found) or 'none'} (put LoRA files into D:\\AI\\models\\comfy\\loras)")
        out.append((hit[0], strength))
    return out


FIXED = {"zimage", "flux3", "fibo2"}      # distilled / CFG-1 workflows: CFG and negative prompt are hard requirements


def model_path(prof):
    """! @brief Where the profile's main model file is (checkpoint or diffusion model). @return path or None """
    folder = "checkpoints" if prof["arch"] == "sdxl" else "diffusion_models"
    return next((f["path"] for f in assets.files(folder) if f["name"] == prof["file"]), None)


def resolve_loras(a, prof, neg, log=print):
    """!
    @brief Settings for a job with LoRAs, in a fixed order: hard requirements of the workflow, then what was typed, then
           the LoRAs' notes (ai gen note), then the model's defaults. Conflicting notes stop the job before the GPU is
           touched; anything overridden is said. Each LoRA's structural fit with the model is checked from the file
           headers: a proven mismatch stops the job (--force-lora runs it anyway, recorded).
    @param a  Parsed arguments.  @param prof  Model profile.  @param neg  Negative prompt so far.
    @return (loras [(file, strength)], prompt, negative, profile, checks [dict for the record])
    """
    prompt, specs, checks = a.prompt, find_loras(a.lora), []
    fixed = prof["arch"] in FIXED
    if a.cfg is not None and not 0.5 <= a.cfg <= 30:
        sys.exit("--cfg must be 0.5-30")
    if a.cfg is not None:
        if fixed:
            log(f"note: --cfg ignored - {prof['name']} runs at its fixed CFG {prof['cfg']}")
        else:
            prof = dict(prof, cfg=a.cfg)
    if fixed and a.negative:
        log(f"note: --negative ignored - {prof['name']} uses no negative prompt")
    mpath, loras, want = model_path(prof), [], {"cfg": {}, "negative": {}}
    for name, strength in specs:
        path = next((os.path.join(d, name) for d in LORA_DIRS if os.path.exists(os.path.join(d, name))), None)
        note = assets.note_for(path) if path else {}
        if strength is None:
            strength = note.get("strength", 1.0)
            if "strength" in note:
                log(f"LoRA {name}: strength {strength} (from its note)")
        trig = note.get("trigger")
        if trig and trig.lower() not in prompt.lower():
            prompt = f"{trig}, {prompt}"
            log(f"LoRA {name}: trigger '{trig}' added to the prompt (from its note)")
        for k in want:
            if k in note:
                want[k][name] = note[k]
        fit = assets.lora_fit(path, mpath) if path and mpath else {"status": "unchecked", "why": "file not found"}
        done = bool(path and mpath and assets.sha256(mpath) in assets.tested(path))
        checks.append({"file": name, "strength": strength, "fit": fit["status"], "why": fit.get("why"),
                       "tested_with_model": done, "note": note or None})
        if fit["status"] == "mismatch" and not a.force_lora:
            sys.exit(f"LoRA {name} does not fit {prof['name']} ({fit['why']}) - it was made for another model type. "
                     "Nothing started. (--force-lora runs it anyway)")
        if fit["status"] != "fits":
            log(f"warning: LoRA {name} with {prof['name']}: {fit['status']} ({fit.get('why')})")
        elif not done:
            log(f"LoRA {name}: fits {prof['name']} ({fit['why']}); not run with this model before")
        loras.append((name, strength))
    for k, typed in (("cfg", a.cfg), ("negative", a.negative)):
        vals = set(map(str, want[k].values()))
        if not vals or fixed or typed is not None:
            if vals and typed is not None and not fixed and vals != {str(typed)}:
                log(f"note: your --{k} is used, the LoRA note recommends {' / '.join(sorted(vals))}")
            continue
        if len(vals) > 1:
            sys.exit(f"the LoRAs' notes disagree on {k}: " + ", ".join(f"{n}: {v!r}" for n, v in want[k].items())
                     + f" - choose one with --{k}. Nothing started.")
        v = next(iter(want[k].values()))
        if k == "cfg":
            prof = dict(prof, cfg=v)
        else:
            neg = v
        log(f"{k} {v!r} (from the LoRA note)")
    return loras, prompt, neg, prof, checks


def apply_loras(wf, loras, arch):
    """!
    @brief Chains LoRA loaders after the model loader: everything that used the base model and text encoder now uses the
           LoRA'd ones (sampler, pose/face nodes, repairs, upscale). SDXL: checkpoint outputs "4"/0,1; the newer models:
           "1" (model) and "2" (text encoder). @param loras  [(file, strength)]  @return the workflow, changed in place
    """
    if not loras:
        return wf
    model, clip = (["4", 0], ["4", 1]) if arch == "sdxl" else (["1", 0], ["2", 0])
    base_model, base_clip = model, clip
    ids = []
    for i, (name, strength) in enumerate(loras):
        nid = f"6{i}"
        wf[nid] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                          "strength_model": strength, "strength_clip": strength}}
        model, clip = [nid, 0], [nid, 1]
        ids.append(nid)
    for nid, node in wf.items():
        if nid in ids:
            continue
        for k, v in node["inputs"].items():
            if v == base_model:
                node["inputs"][k] = model
            elif v == base_clip:
                node["inputs"][k] = clip
    return wf


POSE_CN = "controlnet-union-sdxl-promax.safetensors"
FACE_DET, HAND_DET = "bbox/face_yolov8m.pt", "bbox/hand_yolov8s.pt"


def detailer(image, model, pos, neg, detector, seed, denoise, prof):
    """! @brief FaceDetailer inputs (Impact Pack) that redraw what the detector finds. @return inputs dict """
    return {"image": image, "model": model, "clip": ["4", 1], "vae": ["4", 2], "guide_size": 512, "guide_size_for": True,
            "max_size": 1024, "seed": seed, "steps": 20, "cfg": prof["cfg"], "sampler_name": prof["sampler"],
            "scheduler": prof["scheduler"],
            "positive": pos, "negative": neg, "denoise": denoise, "feather": 5, "noise_mask": True, "force_inpaint": True,
            "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0, "sam_detection_hint": "center-1",
            "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False", "drop_size": 10, "bbox_detector": detector, "wildcard": "", "cycle": 1,
            "inpaint_model": False, "noise_mask_feather": 20, "tiled_encode": False, "tiled_decode": False}


def sdxl(prompt, negative, seed, steps, width, height, upscale, prefix, pose=None, face=None, fix_faces=False,
         fix_hands=False, hires=False, pose_strength=0.65, face_weight=0.65, prof=None, save_base=None):
    """!
    @brief SDXL text-to-image workflow with optional pose control, a face to keep, face/hand repair and upscaling.
    @param prompt    What to draw.  @param negative  What to avoid.  @param seed  Random seed.
    @param steps     Sampling steps.  @param width  Image width.  @param height  Image height.
    @param upscale   Also save a fast 2x copy (4x-UltraSharp).  @param prefix  Output prefix "<date>/<name>".
    @param pose      Uploaded image name: its body pose (DWPose skeleton) steers the picture (union ControlNet).
    @param face      Uploaded image name: this face is kept (IPAdapter FaceID Plus v2, needs InsightFace).
    @param fix_faces Redraw detected faces (FaceDetailer).  @param fix_hands  Redraw detected hands.
    @param hires     Also save a detailed 2x copy (Ultimate SD Upscale: tiles redrawn).
    @param pose_strength  How strictly the pose is followed (0.3-1.0).  @param face_weight  How strongly the face is kept.
    @param prof      Model settings from profiles.json (default model when None).
    @param save_base Also save the image before repair/upscale ("<name>_base"); default: when repairing or upscaling.
    @return Workflow in ComfyUI API format.
    """
    prof = prof or profile(PROFILES.get("default_sdxl", "realvis"))
    if prof["arch"] != "sdxl":
        raise ValueError(f"sdxl() needs an SDXL model, got {prof['name']} ({prof['arch']})")
    S, K, C = prof["sampler"], prof["scheduler"], prof["cfg"]
    wf = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": prof["file"]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative}},
    }
    model, pos, neg = ["4", 0], ["6", 0], ["7", 0]
    if face:
        wf.update({
            "20": {"class_type": "IPAdapterUnifiedLoaderFaceID", "inputs": {"model": model, "preset": "FACEID PLUS V2",
                   "lora_strength": 0.6, "provider": "CUDA"}},
            "21": {"class_type": "LoadImage", "inputs": {"image": face}},
            "22": {"class_type": "IPAdapterFaceID", "inputs": {"model": ["20", 0], "ipadapter": ["20", 1], "image": ["21", 0],
                   "weight": face_weight, "weight_faceidv2": 1.0, "weight_type": "linear", "combine_embeds": "concat",
                   "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
        })
        model = ["22", 0]
    cpos, cneg = pos, neg
    if pose:
        wf.update({
            "30": {"class_type": "LoadImage", "inputs": {"image": pose}},
            "31": _dwpose(["30", 0], "enable"),
            "32": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": POSE_CN}},
            "33": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["32", 0], "type": "openpose"}},
            "34": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": pos, "negative": neg, "control_net": ["33", 0],
                   "image": ["31", 0], "strength": pose_strength, "start_percent": 0.0, "end_percent": 0.75, "vae": ["4", 2]}},
            "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["31", 0]}},
        })
        cpos, cneg = ["34", 0], ["34", 1]
    wf.update({
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": C, "sampler_name": S,
              "scheduler": K, "denoise": 1.0, "model": model, "positive": cpos, "negative": cneg, "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    })
    image = ["8", 0]
    if save_base if save_base is not None else (fix_faces or fix_hands or hires):   # keep the original next to the repaired one
        wf["14"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_base", "images": image}}
    if fix_faces:   # repairs use the plain prompts (no pose steering on the crops) and the face-keeping model
        wf["40"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DET}}
        wf["41"] = {"class_type": "FaceDetailer", "inputs": detailer(image, model, pos, neg, ["40", 0], seed, 0.35, prof)}
        image = ["41", 0]
    if fix_hands:
        wf["42"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": HAND_DET}}
        wf["43"] = {"class_type": "FaceDetailer", "inputs": detailer(image, model, pos, neg, ["42", 0], seed, 0.45, prof)}
        image = ["43", 0]
    wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": image}}
    if upscale or hires:
        wf["10"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.safetensors"}}
    if upscale:
        wf.update({
            "11": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["10", 0], "image": image}},
            "12": {"class_type": "ImageScaleBy", "inputs": {"image": ["11", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
            "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_upscaled", "images": ["12", 0]}},
        })
    if hires:
        wf.update({
            "50": {"class_type": "UltimateSDUpscale", "inputs": {"image": image, "model": model, "positive": pos, "negative": neg,
                   "vae": ["4", 2], "upscale_by": 2.0, "seed": seed, "steps": 20, "cfg": C, "sampler_name": S,
                   "scheduler": K, "denoise": 0.2, "upscale_model": ["10", 0], "mode_type": "Linear", "tile_width": 1024,
                   "tile_height": 1024, "mask_blur": 12, "tile_padding": 48, "seam_fix_mode": "None", "seam_fix_denoise": 1.0,
                   "seam_fix_width": 64, "seam_fix_mask_blur": 8, "seam_fix_padding": 16, "force_uniform_tiles": True,
                   "tiled_decode": False, "batch_size": 1}},
            "51": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_hires", "images": ["50", 0]}},
        })
    return wf


def _loaders(prof, clip_type):
    """! @brief Model, text encoder and VAE loader nodes "1","2","3" for the newer (non-SDXL) architectures. """
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": prof["file"], "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": prof["text_encoder"], "type": clip_type, "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": prof["vae"]}}}


def _finish(wf, prof, seed, steps, model, pos, neg, latent, prefix, upscale):
    """! @brief KSampler + decode + save (+ optional fast 2x copy) shared by the newer architectures. """
    wf.update({
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": prof["cfg"], "sampler_name": prof["sampler"],
              "scheduler": prof["scheduler"], "denoise": 1.0, "model": model, "positive": pos, "negative": neg, "latent_image": latent}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["9", 0]}},
    })
    if upscale:
        wf.update({
            "11": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.safetensors"}},
            "12": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["11", 0], "image": ["9", 0]}},
            "13": {"class_type": "ImageScaleBy", "inputs": {"image": ["12", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
            "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_upscaled", "images": ["13", 0]}},
        })
    return wf


def flux3(prompt, seed, steps, width, height, prefix, prof, refs=(), upscale=False, megapixels=1.0):
    """!
    @brief FLUX 3 Image Klein (distilled) text-to-image, or editing when reference images are given (BFL/Comfy template:
           CFG 1, references through ReferenceMultiLatent, the output takes the first reference's size).
    @param prompt  The full (expanded) caption.  @param refs  Uploaded image names (first = <ref_image_0>).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "flux3")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}})
    if not refs:
        wf["6"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["5", 0], ["6", 0], prefix, upscale)
    rml = {"positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0], "resolution": "resize", "resolution.megapixels": megapixels}
    for i, name in enumerate(refs):
        wf[f"2{i}0"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        rml[f"images.image{i}"] = [f"2{i}0", 0]
    wf["7"] = {"class_type": "ReferenceMultiLatent", "inputs": rml}
    edit = dict(prof, sampler=prof.get("edit_sampler", prof["sampler"]))
    return _finish(wf, edit, seed, steps, ["1", 0], ["7", 0], ["7", 1], ["7", 2], prefix, upscale)


def anima(prompt, negative, seed, steps, width, height, prefix, prof, pose=None, pose_strength=0.8, upscale=False):
    """!
    @brief Anima text-to-image (illustration/anime), optionally steered by a pose (DWPose skeleton -> LLLite pose patch).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "stable_diffusion")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
               "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}})
    model = ["1", 0]
    if pose:
        wf.update({
            "30": {"class_type": "LoadImage", "inputs": {"image": pose}},
            "31": _dwpose(["30", 0]),
            "32": {"class_type": "ModelPatchLoader", "inputs": {"name": prof["pose_patch"]}},
            "33": {"class_type": "AnimaLLLiteApply", "inputs": {"model": model, "model_patch": ["32", 0], "image": ["31", 0],
                   "strength": pose_strength, "start_percent": 0.0, "end_percent": 1.0}},
            "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["31", 0]}},
        })
        model = ["33", 0]
    return _finish(wf, prof, seed, steps, model, ["4", 0], ["5", 0], ["6", 0], prefix, upscale)


def zimage(prompt, seed, steps, width, height, prefix, prof, upscale=False):
    """!
    @brief Z-Image Turbo text-to-image (8 steps, CFG 1, AuraFlow sampling shift 3 - official template).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "lumina2")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
               "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
               "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": prof.get("shift", 3.0)}}})
    return _finish(wf, prof, seed, steps, ["7", 0], ["4", 0], ["5", 0], ["6", 0], prefix, upscale)


def moss(prompt, negative, seed, steps, width, height, prefix, prof, upscale=False):
    """!
    @brief MOSS-Image text-to-image (OpenMOSS; Comfy template image_moss_image_t2i: 25 steps, CFG 2.5, er_sde, simple,
           Qwen3.5-9B text encoder, FLUX.2 small VAE). Good at legible text in images.
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "moss_image")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
               "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}})
    return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["5", 0], ["6", 0], prefix, upscale)


def fibo2(prompt, seed, steps, width, height, prefix, prof, refs=(), upscale=False):
    """!
    @brief Fibo 2.0 (Bria) text-to-image, or editing with up to 4 reference images (Comfy template
           image_fibo2_edit_turbo_pe: TextEncodeFibo2 makes both conditionings and the latent; empty negative = the
           null caption it was trained with). With references the result size is Fibo's trained bucket for the
           first image (about 1 MP); without, width x height.
    @param refs  Uploaded image names (<image_1>, <image_2> ... in the instruction).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "fibo2")
    enc = {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "vae": ["3", 0],
           "width": 0 if refs else width, "height": 0 if refs else height, "batch_size": 1}
    for i, name in enumerate(refs):
        wf[f"2{i}0"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{i + 1}"] = [f"2{i}0", 0]
    wf["4"] = {"class_type": "TextEncodeFibo2", "inputs": enc}
    return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["4", 1], ["4", 2], prefix, upscale)


POSE_PROMPT = ("wide full body photo of a person {d}, entire body from head to feet in frame, space around the figure, "
               "plain light grey studio background, even soft light")


def _dwpose(image, scale_stick="disable"):
    """! @brief DWPose skeleton node (body, hands, face) for an image link. """
    return {"class_type": "DWPreprocessor", "inputs": {"image": image, "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "enable", "resolution": 1024, "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384.onnx", "scale_stick_for_xinsr_cn": scale_stick}}


def pose_wf(description, seed, width, height, prefix, src=None):
    """!
    @brief A pose for ControlNet: from a description, Z-Image makes a plain full-body photo of a person in that pose and
           DWPose turns it into a skeleton; from a photo (src), only the skeleton is made. Saves "<name>" (the photo,
           the best input for --pose) and "<name>_pose" (the skeleton picture).
    @param src  Uploaded photo name, or None to generate one from the description.
    @return Workflow in ComfyUI API format.
    """
    if src:
        return {"30": {"class_type": "LoadImage", "inputs": {"image": src}}, "31": _dwpose(["30", 0]),
                "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["31", 0]}}}
    p = profile("zimage")
    wf = zimage(POSE_PROMPT.format(d=description), seed, p["steps"], width, height, prefix, p)
    wf.update({"31": _dwpose(["9", 0]),
               "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["31", 0]}}})
    return wf


def use_skeleton(wf):
    """!
    @brief The --pose picture is already a skeleton (from ai gen pose or another tool): skip DWPose and feed it
           straight to the pose control (and do not save it again). @return the workflow, changed in place
    """
    if "31" in wf and wf["31"]["class_type"] == "DWPreprocessor":
        del wf["31"]
        wf.pop("35", None)
        for node in wf.values():
            for k, v in node["inputs"].items():
                if v == ["31", 0]:
                    node["inputs"][k] = ["30", 0]
    return wf


def is_skeleton(path, flag=False):
    """! @brief True for a skeleton picture: --pose-skeleton given, or a "<name>_pose_00001_.png" file from ai gen. """
    return flag or re.search(r"_pose_\d+_\.png$", os.path.basename(path)) is not None


EXPANDED = {"ok": None}                      # did the last expand_prompt() really expand? (tests check it)


def model_server_up(log, wait=120):
    """!
    @brief Makes sure the chat-model server (model-serving contract, port LLM_PORT) answers; starts its user service
           (any module unit named in MODEL_SERVICE, default llama-swap) if it does not. @return True when it answers.
    """
    def up():
        try:
            api("/v1/models", timeout=5, base=LLM); return True
        except Exception:
            return False
    if up():
        return True
    unit = env("MODEL_SERVICE", "llama-swap")
    log(f"starting the chat-model server ({unit})")
    subprocess.run(["systemctl", "--user", "start", unit], stderr=subprocess.DEVNULL)
    for _ in range(wait // 3):
        if up():
            return True
        time.sleep(3)
    return False


def expand_prompt(request, kind, width=1024, height=1024, images=(), log=print):
    """!
    @brief FLUX 3 needs a full scene caption: the local chat model (model-serving contract, OpenAI API) expands the
           request with BFL's own upsampler instructions (prompts/flux3_*.txt) before the GPU is handed to ComfyUI.
    @param request  What the user asked.  @param kind  "t2i" or "edit".  @param images  Reference image paths (edit).
    @return The caption, or the request unchanged (with a warning) when the chat model is not available.
    """
    import base64
    sysmsg = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", f"flux3_{kind}.txt")).read()
    g = __import__("math").gcd(width, height)
    text = (f"TARGET CANVAS: {width // g}:{height // g}\n" if kind == "t2i" else "") + request
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": f"data:{IMAGE_TYPES[os.path.splitext(p)[1].lower()]};base64,"
                                                   + base64.b64encode(open(p, "rb").read()).decode()}} for p in images]
    body = {"model": PROFILES.get("expander", "qwen3.5-9b"), "temperature": 0.7, "max_tokens": 1200,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": content}]}
    EXPANDED["ok"] = False
    try:
        if not model_server_up(log):
            raise RuntimeError("the chat-model server did not start (ai logs llama-swap)")
        log("expanding the request into a full caption (local chat model)")
        out = api("/v1/chat/completions", body, timeout=300, base=LLM)["choices"][0]["message"]["content"]
        out = re.sub(r"(?s)<think>.*?</think>", "", out).strip().strip('"')
        if len(out) < 20:
            raise ValueError("empty answer")
        EXPANDED["ok"] = True
        return out
    except Exception as e:
        log(f"CHECK prompt expansion failed ({str(e)[:120]}) - using the request as typed (results will be plainer)")
        return request


def models_report():
    """! @brief Lists every image model in profiles.json and whether all its files are present. @return 0 if no required one is missing """
    roots = [os.path.join(MODELS, "comfy"), os.path.expanduser("~/.local/share/modules/comfyui/models")]
    rc = 0
    for name, p in PROFILES["models"].items():
        need = {"sdxl": [("checkpoints", p["file"])]}.get(p["arch"], [("diffusion_models", p["file"]),
               ("text_encoders", p.get("text_encoder")), ("vae", p.get("vae"))]) + \
            ([("model_patches", p["pose_patch"])] if p.get("pose_patch") else [])
        missing = [f"{d}/{f}" for d, f in need if f and not any(os.path.exists(os.path.join(r, d, f)) for r in roots)]
        state = "ready" if not missing else ("not installed (optional)" if p.get("optional") else "MISSING")
        print(f"  {name:12} {p['arch']:6} {state:26} {p.get('about', '')}")
        for m in missing:
            print(f"               missing {m}")
        if missing and not p.get("optional"):
            rc = 1
    return rc


def wan(prompt, negative, seed, steps, frames, prefix):
    """!
    @brief Wan 2.2 5B (GGUF Q8) text-to-video workflow, 640x352 at 24 fps, saved as animated WebP.
    @param prompt  What happens.  @param negative  What to avoid.  @param seed  Random seed.
    @param steps   Sampling steps.  @param frames  Length (4n+1, e.g. 33 = ~1.4 s, 121 = 5 s).
    @param prefix  Output prefix "<date>/<name>".
    @return Workflow in ComfyUI API format.
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-TI2V-5B-Q8_0.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
              "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 640, "height": 352,
              "length": frames, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 5.0, "sampler_name": "uni_pc",
              "scheduler": "simple", "denoise": 1.0, "model": ["4", 0], "positive": ["5", 0],
              "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["9", 0], "filename_prefix": prefix,
               "fps": 24.0, "lossless": False, "quality": 85, "method": "default"}},
    }


def gpu_used():
    """!
    @brief GPU memory in use. @return MiB, or -1 if nvidia-smi fails.
    """
    try:
        return int(subprocess.check_output([NVSMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return -1


class Peak(threading.Thread):
    """!
    @brief Background thread that records the peak GPU memory (MiB) while a job runs.
    """
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.go = True

    def run(self):
        while self.go:
            self.peak = max(self.peak, gpu_used()); time.sleep(1)


def comfy_up():
    """!
    @brief Tells whether ComfyUI answers. @return True if /system_stats responds.
    """
    try:
        api("/system_stats", timeout=3); return True
    except Exception:
        return False


def unload_llm(log):
    """!
    @brief Unloads every chat model in llama-swap and confirms it: POST /api/models/unload (older builds: GET /unload),
           then waits until /running lists nothing.
    @param log  Progress printer.
    @return True if no chat model is loaded (or the model server is not running), False if one is still loaded.
    """
    try:
        try:
            api("/api/models/unload", {}, base=LLM)
        except urllib.error.HTTPError as e:
            if e.code not in (404, 405):
                raise
            urllib.request.urlopen(f"{LLM}/unload", timeout=30)   # older llama-swap
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), ConnectionRefusedError):
            log("model server not running (nothing to unload)"); return True
        log(f"unload request failed: {e}"); return False
    for _ in range(60):   # llama-swap stops the server asynchronously
        try:
            running = api("/running", base=LLM)["running"]
            if not isinstance(running, list):
                raise ValueError(f"unexpected /running answer: {running!r}")
        except Exception as e:   # unknown or malformed state is not "unloaded"
            log(f"cannot check the model server: {e}"); return False
        if not running:
            log("chat models unloaded"); return True
        time.sleep(1)
    log(f"chat model still loaded after 60 s: {running}"); return False


def comfy_busy(own=None):
    """!
    @brief Tells whether ComfyUI is running or holding jobs that are not ours (e.g. started in the browser).
    @param own  Our prompt id (ignored in the queue), or None.
    @return True if another job is running or waiting.
    """
    q = api("/queue")
    ids = [j[1] for j in q.get("queue_running", []) + q.get("queue_pending", [])]
    return any(i != own for i in ids)


MODEL_EXT = (".safetensors", ".gguf", ".pt", ".pth", ".bin", ".ckpt", ".onnx", ".torchscript")


def missing_files(wf):
    """!
    @brief Preflight: every model file the workflow names must be in the list the loader node offers in the running
           ComfyUI (/object_info - the files ComfyUI itself can see, in every model folder it knows). Runs before the chat
           model is unloaded, so a missing file costs nothing. Image inputs (uploads) are not checked here.
    @param wf  Workflow in API format.  @return list of "Node.input = file" that ComfyUI cannot see
    """
    info, bad = {}, []
    for node in wf.values():
        c = node["class_type"]
        if c not in info:
            try:
                info[c] = api(f"/object_info/{c}").get(c, {}).get("input", {})
            except Exception:
                info[c] = None                     # unknown to ComfyUI: its own validation reports it
        spec = info[c] or {}
        for k, v in node["inputs"].items():
            if not (isinstance(v, str) and v.lower().endswith(MODEL_EXT)):
                continue
            d = spec.get("required", {}).get(k) or spec.get("optional", {}).get(k)
            t = d[0] if isinstance(d, list) and d else None
            opts = t if isinstance(t, list) else (d[1].get("options") if t == "COMBO" and len(d) > 1 and isinstance(d[1], dict) else None)
            if opts is not None and v not in opts:
                bad.append(f"{c}.{k} = {v}")
    return bad


def prepare(log, wait, wf=None):
    """!
    @brief Hands the GPU to ComfyUI: starts it if needed, checks the workflow's model files, waits while ComfyUI runs
           someone else's job, unloads the chat models.
    @param log   Progress printer.
    @param wait  Seconds to wait for a busy ComfyUI.
    @param wf    Workflow to preflight (None: no check).
    @return Error text, or "" when ComfyUI is ready and the GPU is free of chat models.
    """
    if not comfy_up():
        subprocess.run(["systemctl", "--user", "start", "comfyui"], stderr=subprocess.DEVNULL)
        for _ in range(90):
            if comfy_up(): break
            time.sleep(2)
    if not comfy_up():
        return f"ComfyUI is not answering on {URL} (ai logs comfyui)"
    miss = missing_files(wf) if wf else []
    if miss:
        return ("model file(s) not found by ComfyUI - nothing started, the chat model stays loaded: " + "; ".join(miss)
                + " (put them under D:\\AI\\models\\comfy\\<folder>, or: ai mod update comfyui)")
    t0 = time.time()
    while comfy_busy():
        if time.time() - t0 > wait:
            return "ComfyUI is busy with another job (browser?) - try again later"
        time.sleep(5)
    if not unload_llm(log):
        return "the chat model could not be unloaded - not starting (GPU would run out of memory)"
    return ""


def release(log, own=None):
    """!
    @brief Gives the GPU back - ComfyUI unloads its models - unless someone else's job is running or waiting.
    @param log  Progress printer.  @param own  Our prompt id (a just-cancelled job of ours doesn't count as "someone else").
    """
    try:
        if comfy_busy(own):
            log("another ComfyUI job is active - leaving its models loaded"); return
        api("/free", {"unload_models": True, "free_memory": True}); log("ComfyUI GPU memory freed")
    except Exception as e:
        log(f"ComfyUI /free failed: {e}")


def cancel(pid, log):
    """!
    @brief Cancels only our own job: interrupts it if it is the one running, removes it if it is still waiting.
    @param pid  Our prompt id.  @param log  Progress printer.
    """
    q = api("/queue")
    if any(j[1] == pid for j in q.get("queue_running", [])):
        api("/interrupt", {"prompt_id": pid}); log("our job interrupted")   # ComfyUI stops it only if it is still ours
    elif any(j[1] == pid for j in q.get("queue_pending", [])):
        api("/queue", {"delete": [pid]}); log("our job removed from the queue")


def run_job(wf, timeout, log=print):
    """!
    @brief Queues one workflow and waits for it; cancels only this job on timeout.
    @param wf       Workflow in ComfyUI API format.
    @param timeout  Seconds to wait.
    @param log      Progress printer.
    @return (ok, list of output file paths, error text).
    """
    try:
        pid = JOB["pid"] = api("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
        job_state("running", prompt_id=pid)
    except urllib.error.HTTPError as e:
        return False, [], "rejected by ComfyUI: " + e.read().decode()[:800]
    t0 = time.time(); mon = Peak(); mon.start()
    try:
        while time.time() - t0 < timeout:
            time.sleep(3)
            h = api(f"/history/{pid}").get(pid)
            if not h:
                continue
            st = h.get("status", {})
            files = [os.path.join(OUTPUT, f.get("subfolder", ""), f["filename"]) for o in h.get("outputs", {}).values()
                     for k in ("images", "gifs", "videos", "3d") for f in o.get(k, []) if f.get("type") == "output"]
            err = " ".join(m[1].get("exception_message", "")[:400] for m in st.get("messages", []) if m[0] == "execution_error")
            ok = st.get("status_str") == "success" and bool(files) and all(os.path.exists(f) and os.path.getsize(f) > 2000 for f in files)
            JOB.update(secs=round(time.time() - t0), peak_gb=round(mon.peak / 1024, 1))
            log(f"{'done' if ok else 'failed'} in {time.time()-t0:.0f} s, peak GPU memory {mon.peak/1024:.1f} GB")
            return ok, files, err
        cancel(pid, log)
        return False, [], f"timeout after {timeout} s (our job cancelled)"
    finally:
        mon.go = False


JOBS = os.path.join(OUTPUT, "jobs")


def job_state(state, **kv):
    """!
    @brief The job's own small file (~/work/outputs/comfy/jobs/<id>.json), written at every step - waiting, preparing,
           running (with ComfyUI's prompt id), succeeded / failed / cancelled / interrupted / unknown - so a job whose
           client died can still be traced and reconciled. @return the job dict
    """
    j = JOB.setdefault("ledger", {"id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"})
    j.update(kv, state=state, at=datetime.datetime.now().isoformat(timespec="seconds"))
    try:
        os.makedirs(JOBS, exist_ok=True)
        tmp = os.path.join(JOBS, j["id"] + ".tmp")
        with open(tmp, "w") as f:
            json.dump(j, f, indent=1)
        os.replace(tmp, os.path.join(JOBS, j["id"] + ".json"))
    except OSError:
        pass
    return j


def wf_sha(wf):
    """! @brief SHA-256 of a workflow as saved (canonical JSON). """
    import hashlib
    return hashlib.sha256(json.dumps(wf, indent=1).encode()).hexdigest()


def input_hashes(wf, uploads=(), log=print):
    """!
    @brief SHA-256 of every model file the workflow names and every input picture - what a rerun must find unchanged.
           Hashes are cached by path, size and time (the first record for a large model reads it once).
    @return {name: sha256}
    """
    names = {v for n in wf.values() for v in n.get("inputs", {}).values()
             if isinstance(v, str) and v.lower().endswith(MODEL_EXT)}
    where = {f["name"]: f["path"] for f in assets.files() if f["name"] in names}
    cache = assets._load(assets.CACHE)
    if any(p not in cache and os.path.getsize(p) > 1 << 30 for p in where.values()):
        log("recording model file hashes (once per model file, about a minute each; cached afterwards)")
    out = {n: assets.sha256(p) for n, p in sorted(where.items())}
    for _, src in uploads or ():
        if src and os.path.exists(src):
            out["input:" + os.path.basename(src)] = assets.sha256(src)
    return out


def chat_pause(log=print):
    """! @brief Stops the chat-model server while the GPU is busy, so a chat request cannot load a model into a running
           job (llama-swap does not take the GPU lock). @return True when it was running and is now paused """
    if subprocess.run(["systemctl", "--user", "is-active", "-q", "llama-swap"]).returncode != 0:
        return False
    subprocess.run(["systemctl", "--user", "stop", "llama-swap"], stderr=subprocess.DEVNULL)
    log("chat model server paused for the GPU job")
    return True


def chat_resume(paused, log=print):
    """! @brief Starts the chat-model server again when chat_pause() stopped it. """
    if paused:
        subprocess.run(["systemctl", "--user", "start", "llama-swap"], stderr=subprocess.DEVNULL)
        log("chat model server running again")


def generate(wf, timeout, wait=600, log=print, uploads=None):
    """!
    @brief The whole hand-off around one job, under the GPU lock.
    @param wf       Workflow.  @param timeout  Job time limit (s).  @param wait  How long to wait for the lock / a busy ComfyUI (s).
    @param log      Progress printer.  @param uploads  [(node id, image path)]: reference images, handed over once ComfyUI runs.
    @return (ok, output paths, error text).
    """
    JOB.pop("ledger", None)
    job_state("waiting")
    with open(LOCK, "w") as lock:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.time() - t0 > wait:
                    job_state("failed", error="GPU busy (lock held)")
                    return False, [], "GPU busy with another job (lock held) - try again later"
                time.sleep(5)
        job_state("preparing")
        err = prepare(log, max(0, wait - (time.time() - t0)), wf)
        if err:
            job_state("failed", error=err)
            return False, [], err
        paused = chat_pause(log)
        try:
            for node, src in uploads or []:
                wf[node]["inputs"]["image"] = upload(src)
        except Exception as e:
            release(log); chat_resume(paused, log)
            job_state("failed", error=f"upload: {e}")
            return False, [], f"could not hand the reference image to ComfyUI: {e}"
        try:
            ok, files, err = run_job(wf, timeout, log)
            job_state("succeeded" if ok else "cancelled" if err.startswith("timeout") else "failed",
                      error=err or None, outputs=files)
            return ok, files, err
        except KeyboardInterrupt:
            job_state("interrupted", prompt_id=JOB.get("pid"))
            raise
        except Exception as e:                       # lost contact: ComfyUI may still be running it
            job_state("unknown", prompt_id=JOB.get("pid"), error=f"{type(e).__name__}: {e}")
            return False, [], f"lost contact with ComfyUI ({e}) - the job may still finish there (prompt {JOB.get('pid')})"
        finally:
            release(log, JOB.pop("pid", None))
            chat_resume(paused, log)


def all_workflows():
    """! @brief One workflow per model and option combination (for the schema dump and check_workflows.py). """
    import itertools
    for name in PROFILES["models"]:
        p = profile(name)
        if p["arch"] == "sdxl":
            for pose, face, ff, fh, hires, up in itertools.product([None, "p"], [None, "f"], *[[False, True]] * 4):
                yield sdxl("x", "y", 1, p["steps"], 1024, 1024, up, "t", pose=pose, face=face, fix_faces=ff,
                           fix_hands=fh, hires=hires, prof=p)
        elif p["arch"] == "anima":
            for pose, up in itertools.product([None, "p"], [False, True]):
                yield anima("x", "y", 1, p["steps"], 1024, 1024, "t", p, pose=pose, upscale=up)
        elif p["arch"] == "zimage":
            for up in (False, True):
                yield zimage("x", 1, p["steps"], 1024, 1024, "t", p, upscale=up)
        elif p["arch"] == "flux3":
            for refs, up in itertools.product([(), ("a",), ("a", "b")], [False, True]):
                yield flux3("x", 1, p["steps"], 1024, 1024, "t", p, refs=refs, upscale=up)
        elif p["arch"] == "moss":
            for up in (False, True):
                yield moss("x", "y", 1, p["steps"], 1024, 1024, "t", p, upscale=up)
        elif p["arch"] == "fibo2":
            for refs, up in itertools.product([(), ("a",), ("a", "b")], [False, True]):
                yield fibo2("x", 1, p["steps"], 1024, 1024, "t", p, refs=refs, upscale=up)
    for name in PROFILES["models"]:                    # every architecture with two LoRAs chained
        p = profile(name)
        if p.get("kind") == "3d":
            yield apply_recipe(load_recipe(p["workflow"]), dict(p.get("set", {}), image="x.png", prefix="t"))
            continue
        if p["arch"] == "sdxl":
            w = sdxl("x", "y", 1, p["steps"], 1024, 1024, True, "t", pose="p", face="f", fix_faces=True, hires=True, prof=p)
        elif p["arch"] == "anima":
            w = anima("x", "y", 1, p["steps"], 1024, 1024, "t", p, pose="p")
        elif p["arch"] == "zimage":
            w = zimage("x", 1, p["steps"], 1024, 1024, "t", p)
        elif p["arch"] == "flux3":
            w = flux3("x", 1, p["steps"], 1024, 1024, "t", p, refs=("a",))
        elif p["arch"] == "moss":
            w = moss("x", "y", 1, p["steps"], 1024, 1024, "t", p)
        else:
            w = fibo2("x", 1, p["steps"], 1024, 1024, "t", p, refs=("a",))
        yield apply_loras(w, [("a.safetensors", 0.8), ("b.safetensors", -0.5)], p["arch"])
    yield pose_wf("x", 1, 896, 1152, "t")
    yield pose_wf("", 1, 896, 1152, "t", src="a")
    yield use_skeleton(sdxl("x", "y", 1, 30, 1024, 1024, False, "t", pose="p", prof=profile(PROFILES.get("default_sdxl", "realvis"))))
    anima_name = next((n for n, q in PROFILES["models"].items() if q["arch"] == "anima"), None)
    if anima_name:
        yield use_skeleton(anima("x", "y", 1, 30, 1024, 1024, "t", profile(anima_name), pose="p"))
    yield wan("x", "y", 1, 20, 33, "t")


def dump_schema(path):
    """!
    @brief Saves the input definitions of the nodes ai gen uses, read from the running ComfyUI (/object_info), so the
           workflows can be checked without a GPU (check_workflows.py, CI). File-name lists are left out.
    @param path  Where to write the JSON.  @return 0 on success.
    """
    if not comfy_up():
        subprocess.run(["systemctl", "--user", "start", "comfyui"], stderr=subprocess.DEVNULL)
        for _ in range(90):
            if comfy_up(): break
            time.sleep(2)
    info = api("/object_info", timeout=120)
    used = {n["class_type"] for wf in all_workflows() for n in wf.values()}
    out = {}
    for c in sorted(used):
        if c not in info:
            out[c] = None; continue
        spec = {}
        for part in ("required", "optional"):
            for k, v in info[c]["input"].get(part, {}).items():
                t = v[0] if isinstance(v, list) and v else v
                opts = t if isinstance(t, list) else (v[1].get("options") if t in ("COMBO", "COMFY_DYNAMICCOMBO_V3")
                                                      and len(v) > 1 and isinstance(v[1], dict) else None)  # combo formats
                if opts and isinstance(opts[0], dict):           # dynamic combo: the option keys
                    opts = [o.get("key") for o in opts]
                files = opts and any(str(x).lower().endswith((".safetensors", ".pt", ".pth", ".bin", ".gguf", ".onnx",
                                                              ".png", ".jpg", ".jpeg", ".webp")) for x in opts)
                spec[k] = {"part": part, "type": "COMBO" if opts is not None else str(t),
                           "values": opts if opts and not files else None}   # file lists vary per machine
        out[c] = spec
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"schema of {len(out)} nodes written to {path}")
    return 0


def write_record(files, info, wf=None, status="succeeded", error=None, prefix=None, log=print):
    """!
    @brief Saves "<output>.json" next to the main output for every outcome - succeeded, failed, check failed: what was
           asked, what was used (with SHA-256 of every model file and input picture), the job id, how long it took.
           A failed job without outputs is recorded as "<prefix>_failed_<job id>.json".
    @param files  Output paths.  @param info  Request details.  @param wf  The workflow (saved beside it).
    @param status  Outcome.  @param error  Error text.  @param prefix  "<date>/<name>" for a record without outputs.
    @return Path of the record, or None.
    """
    main = [f for f in files if not re.search(r"_(base|pose|hires|upscaled|view_\w+)_?\d*_?\.", os.path.basename(f))] or files[:1]
    job = JOB.get("ledger", {}).get("id", time.strftime("%Y%m%d-%H%M%S"))
    if not main and not prefix:
        return None
    base = os.path.splitext(main[0])[0] if main else os.path.join(OUTPUT, f"{prefix}_failed_{job}")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "status": status, "error": error or None,
           "job": job, **info, "outputs": files, "duration_s": JOB.get("secs"), "peak_gpu_gb": JOB.get("peak_gb")}
    try:
        rec["comfyui_commit"] = subprocess.run(["git", "-C", os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "ComfyUI"),
                                                "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        pass
    if wf is not None:
        try:
            rec["inputs_sha256"] = input_hashes(wf, info.get("uploads"), log)
        except OSError:
            pass
    try:                                                  # module version and checked model hashes, when known
        rec["module"] = open(os.path.expanduser("~/.local/state/modules/comfyui/source")).read().strip()
        hashes = {l.split(" ", 3)[-1].strip(): l.split(" ", 1)[0]
                  for l in open(os.path.expanduser("~/.local/share/modules/comfyui/models/.hashes")) if l.strip()}
        rec["model_sha256"] = {os.path.basename(k): v for k, v in hashes.items() if os.path.basename(k) == info.get("model_file")}
    except OSError:
        pass
    path = base + ".json"
    if wf is not None:              # the exact ComfyUI workflow (API format) as it was run: ai gen rerun, or load it in the page
        wpath = base + ".workflow.json"
        with open(wpath, "w") as f:
            json.dump(wf, f, indent=1)
        rec["workflow"], rec["workflow_sha256"] = os.path.basename(wpath), wf_sha(wf)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return path


HELP = """ai gen - images, edits and short videos on the local GPU (one job at a time; the chat model is unloaded meanwhile)

  ai gen image "prompt" [--name N] [--model M] [--size WxH] [--seed S]   make an image (default model: zimage)
      --style art          artwork/illustration instead of photo defaults      --negative "..."  what to avoid ("" = nothing)
      --pose IMG           copy a body pose (SDXL models, anima)               --pose-strength 0.3-1.0
      --face IMG           keep a face (SDXL models)                           --face-weight 0.3-1.2
      --fix-faces --fix-hands  redraw faces/hands (SDXL)   --hires  detailed 2x copy (SDXL)   --upscale  fast 2x copy
      --checkpoint NAME    any SDXL file in D:\\AI\\models\\comfy\\checkpoints
      --cfg 4.5            guidance (default per model)                       --force-lora  run a LoRA that does not fit
      --lora NAME[:0.8]    a LoRA from D:\\AI\\models\\comfy\\loras (part of its name; strength -2..2, default 1); up to 4
                           - use LoRAs made for the model (SDXL LoRA with SDXL models, FLUX LoRA with flux3 ...)
  ai gen edit "what to change" --image IMG [--image REF]   change an image by instruction (FLUX 3; --model fibo2 = Fibo 2.0)
  ai gen pose "description" | --from PHOTO                 a pose for ControlNet: "<name>" (plain full-body photo,
                                                           best for --pose) + "<name>_pose" (the skeleton picture)
      --pose-skeleton      (image) the --pose picture is already a skeleton (files named *_pose_*.png are detected)
  ai gen video "prompt" [--frames 33]                      Wan 2.2 clip (33 frames ~ 1.4 s, up to 121)
  ai gen 3d --image PHOTO [--model pixal3d|trellis2] [--quality normal|high] [--faces N] [--texture 2048]
                                                           a textured 3D model (GLB) from one picture; Blender renders
                                                           it from 4 sides when installed
  ai gen models                                            which image models are ready
  ai gen list [models|loras|checkpoints] [--json]          what is installed now, with notes and tested pairs
  ai gen show NAME [--json]                                one file: complete?, embedded metadata, note, tested with
  ai gen check --model M --lora L [--lora L2]              does a LoRA fit that model (file headers only, no GPU)
  ai gen note NAME [--trigger T --strength S --cfg C --negative N --base B --about A --source URL]
                   checkpoint: [--sampler S --scheduler K --steps N --cfg C --negative N]
                                                           remember what the LoRA's card says (used automatically)
  ai gen verify                                            re-read files used before: same content?
  ai gen rerun FILE.workflow.json [--name N]               run a saved workflow again, exactly

  --model takes a profile or any model file: ComfyUI's own detection tells what kind it is (from the file's header)
  and it starts from that kind's profile; ai gen note FILE [--sampler --steps --cfg ...] adds its card's settings.
  Every result gets <name>.json (what was asked) and <name>.workflow.json (the exact ComfyUI workflow that ran).
  ai gen schema                                            save ComfyUI node definitions (for the CI check)

  Reference images must be inside ~/work (e.g. ~/work/inputs). Results: ~/work/outputs/comfy/<date>/ (+ a .json record);
  from Windows: dev share. All options: ai gen image --help

Models (--model):"""


def _one(name, folder=None):
    """! @brief A model file by (part of) its name. @return files() entry, or exits listing the candidates """
    fs = [f for f in assets.files(folder) if f["folder"] not in ("insightface",)]
    exact = [f for f in fs if os.path.basename(f["name"]).lower() in (name.lower(), name.lower() + ".safetensors")]
    hit = exact or [f for f in fs if name.lower() in f["name"].lower()]
    if len(hit) != 1:
        sys.exit(f"{name}: {'ambiguous: ' + ', '.join(f['folder'] + '/' + f['name'] for f in hit[:12]) if hit else 'not found'}")
    return hit[0]


def prof_files_all():
    return {p["file"] for p in PROFILES["models"].values()}


def lora_keys_dropped(since):
    """!
    @brief LoRA layers ComfyUI reported it could not apply since a time ("lora key not loaded" / "NOT LOADED" in its log).
    @return count, or None when the log cannot be read
    """
    try:
        r = subprocess.run(["journalctl", "--user", "-u", "comfyui", "--since", f"@{int(since)}", "-o", "cat", "--no-pager"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return sum(1 for l in r.stdout.splitlines() if "lora key not loaded" in l or "NOT LOADED" in l)


QUALITY = {"normal": {"resolution": 1024, "faces": 300000, "texture": 2048},     # ~10 GB peak expected: headroom
           "high": {"resolution": 1536, "faces": 700000, "texture": 4096}}       # the template's own; 10.5 GB measured


def glb_stats(path):
    """! @brief Reads a GLB's header (no extra tools): mesh, triangles, vertices, materials, textures. @return dict """
    with open(path, "rb") as f:
        head = f.read(20)
        if head[:4] != b"glTF":
            return {"valid": False}
        j = json.loads(f.read(int.from_bytes(head[12:16], "little")))
    acc, tri, verts = j.get("accessors", []), 0, 0
    for m in j.get("meshes", []):
        for p in m.get("primitives", []):
            verts += acc[p["attributes"]["POSITION"]]["count"]
            tri += acc[p["indices"]]["count"] // 3 if "indices" in p else 0
    mats = [sorted(k for k in ("baseColorTexture", "metallicRoughnessTexture") if k in m.get("pbrMetallicRoughness", {}))
            + (["normalTexture"] if "normalTexture" in m else []) + (["occlusionTexture"] if "occlusionTexture" in m else [])
            for m in j.get("materials", [])]
    return {"valid": True, "meshes": len(j.get("meshes", [])), "triangles": tri, "vertices": verts, "material_maps": mats,
            "textures": len(j.get("textures", [])), "size_mb": round(os.path.getsize(path) / 1e6, 1)}


def gen3d(a):
    """!
    @brief ai gen 3d --image PHOTO: a textured 3D model (GLB) from one picture, through the image-to-3D recipe (official
           ComfyUI template, core nodes only), with the same GPU hand-off, preflight, record and saved workflow as
           every job; afterwards the GLB is measured and, when the Blender module is installed, rendered from 4 sides
           and image-checked. @return exit code
    """
    if not a.image:
        sys.exit('ai gen 3d needs --image PHOTO (one object, plain background works best; in ~/work)')
    name = a.model or PROFILES.get("default_3d", "pixal3d")
    prof = PROFILES["models"].get(name)
    if not prof or prof.get("kind") != "3d":
        sys.exit(f"--model {name}: 3D models are {', '.join(n for n, p in PROFILES['models'].items() if p.get('kind') == '3d')}")
    src = check_input(a.image[0])
    recipe = load_recipe(prof["workflow"])
    seed = a.seed if a.seed is not None else random.randint(0, 2**31 - 1000)
    day = datetime.date.today().isoformat()
    prefix = f"{day}/{slug(a.name) if a.name else slug(os.path.splitext(os.path.basename(src))[0])}"
    vals = dict(QUALITY[a.quality], **{k: v for k, v in (("faces", a.faces), ("texture", a.texture)) if v is not None})
    vals.update(prof.get("set", {}), seed=seed, prefix=prefix, image="pending", remove_background=not a.keep_background)
    wf = apply_recipe(recipe, vals)
    log = lambda s: print(f"  {s}", flush=True)
    log(f"{name}: {a.quality} quality ({vals['faces']} faces, upsample {vals['resolution']}, texture {vals['texture']}), seed {seed}")
    ok, files, err = generate(wf, a.timeout or 2400, log=log, uploads=[(recipe["inputs"]["image"]["set"][0][0], src)])
    glb = next((f for f in files if f.endswith(".glb")), None)
    info = {"kind": "3d", "model": name, "model_file": prof["file"], "image": a.image[0], "seed": seed, "quality": a.quality,
            **{k: vals[k] for k in ("faces", "resolution", "texture", "remove_background")}, "recipe": prof["workflow"]}
    info["uploads"] = [[recipe["inputs"]["image"]["set"][0][0], src]]
    status, extra = check_3d(glb, info, log) if ok and glb else ("failed", [])
    files += extra
    return report_3d(status, files, info, wf, err, prefix, seed, log)


EXIT = {"succeeded": 0, "failed": 1, "check_failed": 2, "check_skipped": 3}


def check_status(glb_info, blender_present, blender_rc=None, mesh=None, renders=0, imgcheck_rc=None):
    """!
    @brief The verdict on a generated 3D model from its required checks: a valid GLB with triangles; with Blender
           installed, a successful Blender run whose result says ok, 4 renders, and img-check passing all of them.
           Without Blender the model is usable but unchecked ("check_skipped"). @return status (see EXIT)
    """
    if not (glb_info.get("valid") and glb_info.get("triangles", 0) > 0):
        return "failed"
    if not blender_present:
        return "check_skipped"
    if blender_rc != 0 or not isinstance(mesh, dict) or not mesh.get("ok") or renders < 4 or imgcheck_rc != 0:
        return "check_failed"
    return "succeeded"


def check_3d(glb, info, log=print):
    """!
    @brief Required checks of a 3D result: GLB header, then (Blender module) import, mesh measures, 4 renders and
           img-check, with the chat-model server paused (the blender command takes the GPU lock itself).
    @return (status, extra output files)
    """
    info["glb"] = glb_stats(glb)
    blender, views, rc, mesh, chk_rc = shutil.which("blender"), [], None, None, None
    if info["glb"].get("valid") and info["glb"].get("triangles") and blender:
        log("checking the model in Blender (import, mesh, 4 renders)")
        paused = chat_pause(log)
        try:
            r = subprocess.run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_check.py"), "--",
                                glb, os.path.splitext(glb)[0] + "_view"], capture_output=True, text=True, timeout=900)
            rc = r.returncode
            line = next((l for l in r.stdout.splitlines() if l.startswith("MESH-CHECK ")), None)
            mesh = json.loads(line[11:]) if line else {"ok": False, "error": (r.stderr or r.stdout)[-300:]}
        except (subprocess.TimeoutExpired, ValueError) as e:
            mesh = {"ok": False, "error": str(e)[:300]}
        finally:
            chat_resume(paused, log)
        views = [v for v in mesh.get("renders", []) if os.path.exists(v)]
        if views:
            chk = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "img-check"), *views],
                                 capture_output=True, text=True)
            chk_rc, info["renders_check"] = chk.returncode, chk.stdout.strip()[-600:]
        info["blender"] = mesh
    elif not blender:
        info["blender"] = "not installed (ai mod add blender for renders and mesh checks)"
    return check_status(info["glb"], bool(blender), rc, mesh, len(views), chk_rc), views


def report_3d(status, files, info, wf, err, prefix, seed, log=print):
    """! @brief Record and summary for a 3D job; the exit code says whether the required checks passed. """
    why = {"failed": err or "no valid model", "check_failed": "the model was made but a required check failed (see the record)",
           "check_skipped": "the model was made but not checked (Blender module not installed)"}.get(status)
    rec = write_record(files, info, wf, status, why, prefix, log)
    print({"succeeded": f"OUTPUT (seed {seed})", "check_skipped": f"OUTPUT, NOT CHECKED (seed {seed})",
           "check_failed": f"OUTPUT, CHECK FAILED (seed {seed})"}.get(status, f"FAILED {why}"))
    for f in files:
        print(f)
    if status != "failed":
        s = info["glb"]
        b = info.get("blender") if isinstance(info.get("blender"), dict) else {}
        print(f"  model: {s['triangles']} triangles, {s['textures']} textures, {s['size_mb']} MB"
              + (f"; Blender: {b.get('loose_parts')} parts, {b.get('non_manifold_edges')} open edges" if b.get("ok") else ""))
        if status == "check_failed":
            print(f"  check failed: Blender {'ok' if b.get('ok') else b.get('error', 'failed')}; renders: {info.get('renders_check') or 'none'}")
    if rec:
        print(f"  record: {rec}")
    return EXIT[status]


def asset_cmd(argv):
    """!
    @brief ai gen list | show | check | note | verify | names - what is installed and what is known about it. Lists the
           folders as they are now (nothing to keep in sync); headers are read only for show and check.
    @param argv  Arguments after "ai gen".  @return exit code
    """
    ap = argparse.ArgumentParser(prog="ai gen " + argv[0])
    cmd = argv[0]
    if cmd == "list":
        ap.add_argument("what", nargs="?", choices=["models", "loras", "checkpoints", "all"], default="all")
    elif cmd in ("show", "note"):
        ap.add_argument("name")
    elif cmd == "names":
        ap.add_argument("what", choices=["models", "loras", "checkpoints"])
    if cmd == "check":
        ap.add_argument("--model"); ap.add_argument("--checkpoint"); ap.add_argument("--lora", action="append", default=[])
    if cmd == "note":
        for k in assets.NOTE_KEYS:
            ap.add_argument("--" + k, default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output (for agents)")
    a = ap.parse_args(argv[1:])
    out = {}
    if cmd == "names":
        names = (list(PROFILES["models"]) + [os.path.splitext(f["name"])[0] for d in ("checkpoints", "diffusion_models")
                                             for f in assets.files(d) if f["name"] not in prof_files_all()]
                 if a.what == "models" else
                 [f["name"] for f in assets.files(a.what if a.what == "loras" else "checkpoints")])
        print("\n".join(names)); return 0
    if cmd == "list":
        prof_files = {p["file"]: n for n, p in PROFILES["models"].items()}
        if a.what in ("models", "all"):
            on_disk = {f["name"] for f in assets.files("checkpoints")} | {f["name"] for f in assets.files("diffusion_models")}
            out["models"] = [{"name": n, "arch": p["arch"], "ready": p["file"] in on_disk, "about": p.get("about", "")}
                             for n, p in PROFILES["models"].items()]
        if a.what in ("checkpoints", "all"):        # every model file: how to use it (one rule for all kinds)
            cache, allnotes, detected = assets._load(assets.CACHE), assets.notes(), assets._load(DETECT_CACHE)
            ca = class_arch() if detected else {}
            out["checkpoints"] = []
            for d in ("checkpoints", "diffusion_models"):
                for f in assets.files(d):
                    if not f["name"].endswith((".safetensors", ".gguf")):
                        continue
                    sha = (cache.get(f["path"]) or [0, 0, None])[2]
                    arch = allnotes.get(sha, {}).get("arch") if sha else None
                    dc = (detected.get(f["path"]) or [0, 0, None])[2]      # ComfyUI's detection, when already run
                    kind = arch or ca.get(dc)
                    p3 = PROFILES["models"].get(prof_files.get(f["name"]), {}).get("kind") == "3d"
                    use = (f"ai gen 3d --model {prof_files[f['name']]}" if p3 else
                           f"--model {prof_files[f['name']]}" if f["name"] in prof_files else
                           f"--model {os.path.splitext(f['name'])[0]} ({kind})" if kind else
                           f"ComfyUI: {dc} - no ai gen workflow for that kind yet" if dc else
                           f"--model {os.path.splitext(f['name'])[0]} (kind checked on first use)" if f["name"].endswith(".safetensors")
                           else "GGUF: used by the video workflow, not by --model")
                    out["checkpoints"].append({"file": f["name"], "folder": d, "profile": prof_files.get(f["name"]), "use": use})
        if a.what in ("loras", "all"):
            allnotes, alltested = assets.notes(), assets.tested()
            cache = assets._load(assets.CACHE)
            out["loras"] = []
            for f in assets.files("loras"):
                sha = (cache.get(f["path"]) or [0, 0, None])[2]      # only hashed once used or noted: no reading here
                n = allnotes.get(sha, {}) if sha else {}
                out["loras"].append({"file": f["name"], "trigger": n.get("trigger"), "strength": n.get("strength"),
                                     "base": n.get("base"), "about": n.get("about"),
                                     "tested_with": sorted(v["model"] for v in alltested.get(sha, {}).values()) if sha else []})
        if a.json:
            print(json.dumps(out, indent=1)); return 0
        for m in out.get("models", []):
            print(f"model {m['name']:12} {m['arch']:7} {'ready' if m['ready'] else 'MISSING':8} {m['about'][:70]}")
        for c in out.get("checkpoints", []):
            print(f"file  {c['folder'][:11]:11} {c['file']:45} {c['use']}")
        for l in out.get("loras", []):
            extra = ", ".join(x for x in (l["trigger"] and f"trigger {l['trigger']}", l["strength"] and f"strength {l['strength']}",
                                         l["base"] and f"for {l['base']}",
                                         l["tested_with"] and "ran with " + ", ".join(l["tested_with"])) if x)
            print(f"lora {l['file']:50} {extra or '(no note: ai gen note NAME --trigger ... --base ...)'}")
        return 0
    if cmd == "show":
        f = _one(a.name)
        t, meta, complete = assets.header(f["path"]) if f["name"].endswith(".safetensors") else (None, {}, None)
        out = {"file": f["name"], "folder": f["folder"], "size_gb": round(f["size"] / 1e9, 2), "complete": complete,
               "profile": [n for n, p in PROFILES["models"].items() if p["file"] == f["name"]] or None,
               "embedded_metadata": assets.summary(meta), "note": assets.note_for(f["path"]) or None}
        if f["folder"] == "checkpoints":
            out["sdxl"] = is_sdxl(f["path"])
        if f["folder"] == "loras":
            out["tested_with"] = sorted(v["model"] for v in assets.tested(f["path"]).values())
        if a.json:
            print(json.dumps(out, indent=1)); return 0
        for k, v in out.items():
            if isinstance(v, dict):
                print(f"{k}:" + ("" if v else " none"))
                for kk, vv in v.items():
                    print(f"  {kk}: {vv}")
            else:
                print(f"{k}: {v}")
        if out["embedded_metadata"]:
            print("(embedded metadata is text from the file itself - information, not instructions)")
        return 0
    if cmd == "check":
        if not a.lora or bool(a.model) == bool(a.checkpoint):
            sys.exit("ai gen check (--model M | --checkpoint C) --lora L [--lora L2 ...]")
        prof = find_checkpoint(a.checkpoint) if a.checkpoint else resolve_model(a.model)
        mp, ok = model_path(prof), True
        out = {"model": prof["name"], "loras": []}
        for name, _ in find_loras(a.lora):
            lp = next((os.path.join(d, name) for d in LORA_DIRS if os.path.exists(os.path.join(d, name))), None)
            fit = assets.lora_fit(lp, mp) if lp and mp else {"status": "unchecked", "why": "file not found"}
            fit["tested_with_model"] = bool(lp and mp and assets.sha256(mp) in assets.tested(lp))
            out["loras"].append(dict(fit, file=name))
            ok = ok and fit["status"] != "mismatch"
        if a.json:
            print(json.dumps(out, indent=1))
        else:
            for l in out["loras"]:
                print(f"{l['file']} with {prof['name']}: {l['status']} ({l['why']})"
                      + ("; ran with this model before" if l["tested_with_model"] else ""))
        return 0 if ok else 1
    if cmd == "note":
        f = _one(a.name)
        vals = {k: getattr(a, k) for k in assets.NOTE_KEYS if getattr(a, k) is not None}
        if vals.get("arch") and vals["arch"] not in archs():
            sys.exit(f"--arch {vals['arch']}: ai gen has workflows for {', '.join(sorted(archs()))}")
        for k, d in (("text_encoder", "text_encoders"), ("vae", "vae")):
            if vals.get(k) and not any(x["name"] == vals[k] for x in assets.files(d)):
                sys.exit(f"--{k} {vals[k]}: not in {d}/ (ai gen list)")
        try:                                   # sampler/scheduler names the pinned ComfyUI knows (schema.json)
            ks = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")))["KSampler"]
            for k, inp in (("sampler", "sampler_name"), ("scheduler", "scheduler")):
                if vals.get(k) and vals[k] not in ks[inp]["values"]:
                    sys.exit(f"--{k} {vals[k]}: not known to ComfyUI. Choose from: {', '.join(ks[inp]['values'])}")
        except (OSError, KeyError, ValueError):
            pass
        if "steps" in vals and vals["steps"] != "" and not 1 <= int(vals["steps"]) <= 60:
            sys.exit("--steps must be 1-60")
        try:
            n = assets.set_note(f["path"], vals) if vals else assets.note_for(f["path"])
        except ValueError as e:
            sys.exit(f"note: {e}")
        print(json.dumps(n, indent=1) if n else f"no note for {f['name']} (ai gen note {a.name} --trigger ... --strength ...)")
        return 0
    if cmd == "verify":
        bad = assets.verify()
        for p, why in bad:
            print(f"CHANGED {p}: {why}")
        print("every file used before still has the same content" if not bad else f"{len(bad)} changed")
        return 1 if bad else 0
    return 2


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
    a = ap.parse_args(argv)
    p = os.path.realpath(os.path.expanduser(a.workflow))
    if not p.startswith(WORK + os.sep) or not p.endswith(".workflow.json"):
        sys.exit("ai gen rerun: a .workflow.json inside ~/work (e.g. ~/work/outputs/comfy/<date>/x_00001_.workflow.json)")
    recp = p[: -len(".workflow.json")] + ".json"
    try:
        old = json.load(open(recp))
    except (OSError, ValueError):
        sys.exit(f"ai gen rerun: no record {os.path.basename(recp)} beside it - only workflows saved by ai gen can be rerun")
    wf = json.load(open(p))
    if old.get("workflow") != os.path.basename(p) or old.get("workflow_sha256") != wf_sha(wf):
        sys.exit("ai gen rerun: the workflow does not match its record (changed since, or saved before records had hashes)")
    try:
        known = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json"))))
    except (OSError, ValueError):
        known = set()
    odd = sorted({n["class_type"] for n in wf.values()} - known)
    if odd:
        sys.exit(f"ai gen rerun: node types ai gen does not use: {', '.join(odd)} - not run")
    uploads = [(n, s) for n, s in old.get("uploads", [])]
    log = lambda s: print(f"  {s}", flush=True)
    now = input_hashes(wf, uploads, log)
    drift = sorted(k for k in set(now) | set(old.get("inputs_sha256", {})) if now.get(k) != old.get("inputs_sha256", {}).get(k))
    if drift and not a.allow_drift:
        sys.exit(f"ai gen rerun: changed or missing since the original: {', '.join(drift)} (--allow-drift runs anyway)")
    day = datetime.date.today().isoformat()
    prefix = f"{day}/{slug(a.name) if a.name else 'rerun-' + slug(os.path.basename(p).split('_0')[0])}"
    for node in wf.values():                       # outputs go to today's folder under the new name, nothing else changes
        if node.get("class_type", "").startswith("Save") and "filename_prefix" in node.get("inputs", {}):
            oldp = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = prefix + (oldp[oldp.rfind("_"):] if re.search(r"_(base|pose|hires|upscaled)$", oldp) else "")
    ok, files, err = generate(wf, a.timeout, log=log, uploads=uploads)
    info = {k: v for k, v in old.items() if k not in ("time", "status", "error", "job", "outputs", "duration_s", "peak_gpu_gb",
                                                       "workflow", "workflow_sha256", "inputs_sha256", "glb", "blender", "renders_check")}
    info.update(rerun_of=p, drift=drift or None)
    if old.get("kind") == "3d":
        glb = next((f for f in files if f.endswith(".glb")), None)
        status, extra = check_3d(glb, info, log) if ok and glb else ("failed", [])
        return report_3d(status, files + extra, info, wf, err, prefix, old.get("seed", 0), log)
    rec = write_record(files, info, wf, "succeeded" if ok else "failed", err, prefix, log)
    print("OUTPUT" if ok else f"FAILED {err}")
    for f in files:
        print(f)
    if rec:
        print(f"  record: {rec}")
    return 0 if ok else 1


def main():
    """!
    @brief Command-line entry point. @return 0 on success, 1 on failure.
    """
    if sys.argv[1:2] == ["rerun"]:
        return rerun(sys.argv[2:])
    if sys.argv[1:2] and sys.argv[1] in ("list", "show", "check", "note", "verify", "names"):
        return asset_cmd(sys.argv[1:])
    ap = argparse.ArgumentParser(description="Generate an image or video on the GPU (hands the GPU over from the chat model).")
    if sys.argv[1:2] == ["schema"]:              # ai gen schema: node definitions for the offline workflow check
        return dump_schema(os.path.join(OUTPUT, "schema.json"))
    if sys.argv[1:2] == ["models"]:              # ai gen models: which image models are ready
        return models_report()
    if not sys.argv[1:] or sys.argv[1] in ("help", "-h", "--help") or sys.argv[2:3] == ["help"]:   # "image --help": all options
        print(HELP); models_report(); return 0
    ap.add_argument("kind", choices=["image", "video", "edit", "pose", "3d"],
                    help="edit = change an image by instruction; pose = a ControlNet pose (photo + skeleton)")
    ap.add_argument("prompt", nargs="?", default="", help="what to make; for edit: what to change; for pose: the pose")
    ap.add_argument("--from", dest="src", metavar="PHOTO", help="pose: make the skeleton from this photo (in ~/work)")
    ap.add_argument("--pose-skeleton", action="store_true", help="image: the --pose picture is already a skeleton")
    ap.add_argument("--image", metavar="IMAGE", action="append", default=[],
                    help="edit: the image to change (in ~/work); repeat for extra references (up to 4)")
    ap.add_argument("--no-expand", action="store_true", help="FLUX 3: use the prompt as typed (no caption expansion)")
    ap.add_argument("--name", default="", help="short name for the output files (default: from the prompt)")
    ap.add_argument("--model", metavar="NAME", help=f"image model: a profile ({', '.join(PROFILES['models'])}; default {PROFILES['default']}) or any model file (ai gen list)")
    ap.add_argument("--style", choices=list(PROFILES["styles"]), default=None, help="image: photo or art (default: per model)")
    ap.add_argument("--negative", default=None, help='what to avoid (default: per model and style; "" = nothing)')
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
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
    a = ap.parse_args()
    if a.kind == "3d":
        return gen3d(a)
    # limits for a 12 GB GPU (and against a runaway agent): larger jobs fail or block the GPU for hours
    if a.steps is not None and not 1 <= a.steps <= MAX_STEPS:
        sys.exit(f"--steps must be 1-{MAX_STEPS}")
    if a.timeout is not None and not 60 <= a.timeout <= MAX_TIMEOUT:
        sys.exit(f"--timeout must be 60-{MAX_TIMEOUT} s")
    if a.kind == "video" and not 5 <= a.frames <= MAX_FRAMES:
        sys.exit(f"--frames must be 5-{MAX_FRAMES}")
    if not a.prompt.strip() and not (a.kind == "pose" and a.src):
        sys.exit(f'ai gen {a.kind} needs a prompt, e.g. ai gen {a.kind} "..."' + (" (or --from PHOTO)" if a.kind == "pose" else ""))
    if a.src and a.kind != "pose":
        sys.exit("--from is for ai gen pose (use --pose to copy a pose into an image)")
    a.size = a.size or ("896x1152" if a.kind == "pose" else "1024x1024")
    os.makedirs(OUTPUT, exist_ok=True)
    if shutil.disk_usage(OUTPUT).free < MIN_FREE:
        sys.exit(f"less than {MIN_FREE >> 30} GB free in {OUTPUT}")
    seed = a.seed if a.seed is not None else random.randint(1, 2**31)
    prefix = f"{datetime.date.today():%Y-%m-%d}/{slug(a.name or ' '.join(a.prompt.split()[:6]) or 'pose')}"
    if a.kind == "pose":
        m = re.fullmatch(r"(\d{3,4})x(\d{3,4})", a.size)
        if not m:
            sys.exit("--size must look like 896x1152")
        w, h = (int(v) // 64 * 64 for v in m.groups())
        if not (512 <= w <= MAX_SIDE and 512 <= h <= MAX_SIDE):
            sys.exit(f"--size: each side must be 512-{MAX_SIDE}")
        if a.pose or a.face or a.fix_faces or a.fix_hands or a.hires or a.checkpoint or a.model or a.image or a.lora:
            sys.exit("ai gen pose takes a description or --from PHOTO, plus --name/--seed/--size")
        uploads = [("30", check_input(a.src))] if a.src else []
        wf = pose_wf(a.prompt, seed, w, h, prefix, src="pending" if a.src else None)
        info = {"kind": "pose", "prompt": a.prompt or None, "full_prompt": None if a.src else POSE_PROMPT.format(d=a.prompt),
                "from": a.src, "seed": None if a.src else seed, "model": None if a.src else "zimage", "size": f"{w}x{h}",
                "use": "ai gen image \"...\" --pose <the photo, or the _pose skeleton>"}
        limit = a.timeout or 900
    elif a.kind in ("image", "edit"):
        m = re.fullmatch(r"(\d{3,4})x(\d{3,4})", a.size)
        if not m:
            sys.exit("--size must look like 1024x1024")
        w, h = (int(v) // 64 * 64 for v in m.groups())
        if not (512 <= w <= MAX_SIDE and 512 <= h <= MAX_SIDE):
            sys.exit(f"--size: each side must be 512-{MAX_SIDE}")
        if a.model and a.checkpoint:
            sys.exit("use --model or --checkpoint, not both")
        if a.kind == "edit":
            if not a.image or len(a.image) > 4:
                sys.exit("edit needs --image <file> (up to 4 with extra references)")
            a.model = a.model or PROFILES.get("edit_model", "flux3")
        if not a.model and not a.checkpoint and (a.pose or a.face or a.fix_faces or a.fix_hands or a.hires):
            a.model = PROFILES.get("default_sdxl", "realvis")    # those tools are SDXL-only
        log0 = lambda t: print(f"  {t}", flush=True)
        prof = find_checkpoint(a.checkpoint, log0) if a.checkpoint else resolve_model(a.model or PROFILES["default"], log0)
        arch, style = prof["arch"], a.style or prof.get("style", "photo")
        allowed = {"sdxl": {"pose", "face", "fix_faces", "fix_hands", "hires"}, "anima": {"pose"}}.get(arch, set())
        used = [o for o in ("pose", "face", "fix_faces", "fix_hands", "hires") if getattr(a, o)]
        if [o for o in used if o not in allowed]:
            sys.exit(f"--model {prof['name']} ({arch}) does not support --{', --'.join(o.replace('_', '-') for o in used if o not in allowed)}"
                     " - those work with the SDXL models (realvis, juggernaut, sdxl-base)" + (" and --pose with anima" if arch != "anima" else ""))
        if a.kind == "edit" and arch not in ("flux3", "fibo2"):
            sys.exit("edit needs a FLUX 3 or Fibo 2.0 model (--model flux3 | fibo2)")
        if a.image and a.kind != "edit":
            sys.exit("--image is for edit")
        pose_strength = a.pose_strength if a.pose_strength is not None else prof.get("pose_strength", 0.65)
        if not (0.3 <= pose_strength <= 1.0 and 0.3 <= a.face_weight <= 1.2):
            sys.exit("--pose-strength must be 0.3-1.0 and --face-weight 0.3-1.2")
        neg = a.negative if a.negative is not None else negative_for(prof, style)
        steps = a.steps or prof["steps"]
        loras, prompt, neg, prof, lora_checks = resolve_loras(a, prof, neg, log0)
        if arch == "sdxl":
            uploads = [(n, check_input(f)) for n, f in (("30", a.pose), ("21", a.face)) if f]
            wf = sdxl(prompt, neg, seed, steps, w, h, a.upscale, prefix, pose="pending" if a.pose else None,
                      face="pending" if a.face else None, fix_faces=a.fix_faces, fix_hands=a.fix_hands, hires=a.hires,
                      pose_strength=pose_strength, face_weight=a.face_weight, prof=prof)
            if a.pose and is_skeleton(a.pose, a.pose_skeleton):
                use_skeleton(wf)
        elif arch == "anima":
            uploads = [("30", check_input(a.pose))] if a.pose else []
            wf = anima(prompt, neg, seed, steps, w, h, prefix, prof, pose="pending" if a.pose else None,
                       pose_strength=pose_strength, upscale=a.upscale)
            if a.pose and is_skeleton(a.pose, a.pose_skeleton):
                use_skeleton(wf)
        elif arch == "zimage":
            uploads, neg = [], ""                           # CFG 1: no negative prompt
            wf = zimage(prompt, seed, steps, w, h, prefix, prof, upscale=a.upscale)
        elif arch == "flux3":
            refs = [check_input(f) for f in a.image]
            if not a.no_expand:
                prompt = expand_prompt(a.prompt, "edit" if refs else "t2i", w, h, refs, log=log0)
            uploads, neg = [(f"2{i}0", f) for i, f in enumerate(refs)], ""
            wf = flux3(prompt, seed, steps, w, h, prefix, prof, refs=["pending"] * len(refs), upscale=a.upscale)
        elif arch == "moss":
            uploads = []
            wf = moss(prompt, neg, seed, steps, w, h, prefix, prof, upscale=a.upscale)
        elif arch == "fibo2":
            refs = [check_input(f) for f in a.image]
            uploads, neg = [(f"2{i}0", f) for i, f in enumerate(refs)], ""
            wf = fibo2(prompt, seed, steps, w, h, prefix, prof, refs=["pending"] * len(refs), upscale=a.upscale)
        else:
            sys.exit(f"--model {prof['name']}: architecture {arch} is not wired into ai gen")
        apply_loras(wf, loras, arch)
        info = {"kind": a.kind, "prompt": a.prompt, "expanded_prompt": prompt if prompt != a.prompt else None,
                "negative": neg, "seed": seed, "model": prof["name"], "arch": arch,
                "model_file": prof["file"], "sampler": prof["sampler"], "scheduler": prof["scheduler"], "cfg": prof["cfg"],
                "steps": steps, "size": f"{w}x{h}", "style": style,
                "options": {k: getattr(a, k) for k in ("upscale", "hires", "fix_faces", "fix_hands")},
                "pose": a.pose and {"file": a.pose, "strength": pose_strength},
                "face": a.face and {"file": a.face, "weight": a.face_weight}, "images": a.image or None,
                "loras": lora_checks or None}
        limit = a.timeout or (1800 if a.hires or a.fix_hands or a.fix_faces or arch in ("flux3", "fibo2", "moss") else 900)
    else:
        frames = max(5, (a.frames - 1) // 4 * 4 + 1)
        if a.lora or a.pose_skeleton or a.pose or a.face or a.fix_faces or a.fix_hands or a.hires or a.checkpoint or a.model or a.image:
            sys.exit("--pose/--face/--fix-faces/--fix-hands/--hires/--checkpoint/--model/--image are for images")
        uploads = []
        neg = a.negative if a.negative is not None else "static, blurry, distorted, low quality, watermark"
        wf, limit = wan(a.prompt, neg, seed, a.steps or 20, frames, prefix), a.timeout or 3600
        info = {"kind": "video", "prompt": a.prompt, "negative": neg, "seed": seed, "model": "wan2.2-ti2v-5b",
                "steps": a.steps or 20, "frames": frames}
    log = lambda s: print(f"  {s}", flush=True)
    t_start = time.time()
    ok, files, err = generate(wf, limit, log=log, uploads=uploads)
    if not ok and re.search(r"read_file_slice|HostBuffer|deserializ|Unable to load|out of memory", err or "", re.I):
        # model file read from D: failed (seen right after the chat model unloaded, and mid-job after a pose preview
        # was already saved - so earlier files do not mean the job got past loading): retry the whole job once
        log(f"model loading failed ({err[:80]}) - retrying once")
        ok, files, err = generate(wf, limit, log=log, uploads=uploads)
    if err:
        print(f"  error: {err}")
    info["uploads"] = [[n, s] for n, s in uploads]
    rec = write_record(files, info, wf, "succeeded" if ok else "failed", err, prefix)
    if ok and a.kind != "video" and info.get("loras"):       # ComfyUI's verdict: did it apply every LoRA layer?
        dropped = lora_keys_dropped(t_start)
        if dropped:
            print(f"  warning: ComfyUI could not apply {dropped} LoRA layer(s) in this job (ai logs comfyui) - "
                  "not recorded as tested")
        elif dropped == 0:
            mp = model_path(prof)
            for d in LORA_DIRS:
                for c in info["loras"]:
                    if mp and os.path.exists(os.path.join(d, c["file"])):
                        assets.record_tested(os.path.join(d, c["file"]), mp)
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
    sys.exit(main())
