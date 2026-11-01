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
CKPT_DIRS = [os.path.join(env("MODELS_MOUNT", "/models"), "comfy", "checkpoints"),
             os.path.expanduser("~/.local/share/modules/comfyui/models/checkpoints")]


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


def find_checkpoint(name):
    """!
    @brief --checkpoint: an SDXL checkpoint by (part of) its file name. Uses the matching profile's settings when there is
           one, else the default model's settings.
    @param name  Name or part of it (case-insensitive).  @return Profile dict with "file" set, or exits listing what is there.
    """
    found = {f: os.path.join(d, f) for d in CKPT_DIRS if os.path.isdir(d) for f in os.listdir(d) if f.endswith(".safetensors")}
    hit = [f for f in sorted(found) if name.lower() in f.lower()]
    if len(hit) != 1:
        sys.exit(f"--checkpoint {name}: {'ambiguous' if hit else 'not found'}. Available: {', '.join(sorted(found)) or 'none'}"
                 " (add your own to D:\\AI\\models\\comfy\\checkpoints)")
    if not is_sdxl(found[hit[0]]):
        sys.exit(f"--checkpoint {name}: {hit[0]} is not an SDXL checkpoint - this workflow needs SDXL")
    known = [k for k, v in PROFILES["models"].items() if v.get("file") == hit[0]]
    return profile(known[0]) if known else dict(profile(), file=hit[0], name=hit[0])


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
            "31": {"class_type": "DWPreprocessor", "inputs": {"image": ["30", 0], "detect_hand": "enable", "detect_body": "enable",
                   "detect_face": "enable", "resolution": 1024, "bbox_detector": "yolox_l.onnx",
                   "pose_estimator": "dw-ll_ucoco_384.onnx", "scale_stick_for_xinsr_cn": "enable"}},
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
            "31": {"class_type": "DWPreprocessor", "inputs": {"image": ["30", 0], "detect_hand": "enable", "detect_body": "enable",
                   "detect_face": "enable", "resolution": 1024, "bbox_detector": "yolox_l.onnx",
                   "pose_estimator": "dw-ll_ucoco_384.onnx", "scale_stick_for_xinsr_cn": "disable"}},
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
    roots = [os.path.join(env("MODELS_MOUNT", "/models"), "comfy"), os.path.expanduser("~/.local/share/modules/comfyui/models")]
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


def prepare(log, wait):
    """!
    @brief Hands the GPU to ComfyUI: starts it if needed, waits while it runs someone else's job, unloads the chat models.
    @param log   Progress printer.
    @param wait  Seconds to wait for a busy ComfyUI.
    @return Error text, or "" when ComfyUI is ready and the GPU is free of chat models.
    """
    if not comfy_up():
        subprocess.run(["systemctl", "--user", "start", "comfyui"], stderr=subprocess.DEVNULL)
        for _ in range(90):
            if comfy_up(): break
            time.sleep(2)
    if not comfy_up():
        return f"ComfyUI is not answering on {URL} (ai logs comfyui)"
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
                     for k in ("images", "gifs", "videos") for f in o.get(k, []) if f.get("type") == "output"]
            err = " ".join(m[1].get("exception_message", "")[:400] for m in st.get("messages", []) if m[0] == "execution_error")
            ok = st.get("status_str") == "success" and bool(files) and all(os.path.exists(f) and os.path.getsize(f) > 2000 for f in files)
            JOB.update(secs=round(time.time() - t0), peak_gb=round(mon.peak / 1024, 1))
            log(f"{'done' if ok else 'failed'} in {time.time()-t0:.0f} s, peak GPU memory {mon.peak/1024:.1f} GB")
            return ok, files, err
        cancel(pid, log)
        return False, [], f"timeout after {timeout} s (our job cancelled)"
    finally:
        mon.go = False


def generate(wf, timeout, wait=600, log=print, uploads=None):
    """!
    @brief The whole hand-off around one job, under the GPU lock.
    @param wf       Workflow.  @param timeout  Job time limit (s).  @param wait  How long to wait for the lock / a busy ComfyUI (s).
    @param log      Progress printer.  @param uploads  [(node id, image path)]: reference images, handed over once ComfyUI runs.
    @return (ok, output paths, error text).
    """
    with open(LOCK, "w") as lock:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.time() - t0 > wait:
                    return False, [], "GPU busy with another job (lock held) - try again later"
                time.sleep(5)
        err = prepare(log, max(0, wait - (time.time() - t0)))
        if err:
            return False, [], err
        try:
            for node, src in uploads or []:
                wf[node]["inputs"]["image"] = upload(src)
        except Exception as e:
            release(log)
            return False, [], f"could not hand the reference image to ComfyUI: {e}"
        try:
            return run_job(wf, timeout, log)
        finally:
            release(log, JOB.pop("pid", None))


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


def write_record(files, info):
    """!
    @brief Saves "<image>.json" next to the main output: what was asked, what was used, how long it took.
    @param files  Output paths.  @param info  Request details.  @return Path of the record, or None.
    """
    main = [f for f in files if not re.search(r"_(base|pose|hires|upscaled)_\d+_\.", os.path.basename(f))] or files[:1]
    if not main:
        return None
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **info, "outputs": files,
           "duration_s": JOB.get("secs"), "peak_gpu_gb": JOB.get("peak_gb")}
    try:                                                  # module version and checked model hashes, when known
        rec["module"] = open(os.path.expanduser("~/.local/state/modules/comfyui/source")).read().strip()
        hashes = {l.split(" ", 3)[-1].strip(): l.split(" ", 1)[0]
                  for l in open(os.path.expanduser("~/.local/share/modules/comfyui/models/.hashes")) if l.strip()}
        rec["model_sha256"] = {os.path.basename(k): v for k, v in hashes.items() if os.path.basename(k) == info.get("model_file")}
    except OSError:
        pass
    path = os.path.splitext(main[0])[0] + ".json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return path


def main():
    """!
    @brief Command-line entry point. @return 0 on success, 1 on failure.
    """
    ap = argparse.ArgumentParser(description="Generate an image or video on the GPU (hands the GPU over from the chat model).")
    if sys.argv[1:2] == ["schema"]:              # ai gen schema: node definitions for the offline workflow check
        return dump_schema(os.path.join(OUTPUT, "schema.json"))
    if sys.argv[1:2] == ["models"]:              # ai gen models: which image models are ready
        return models_report()
    ap.add_argument("kind", choices=["image", "video", "edit"], help="edit = change an image by instruction (FLUX 3)")
    ap.add_argument("prompt", help="what to make; for edit: what to change")
    ap.add_argument("--image", metavar="IMAGE", action="append", default=[],
                    help="edit: the image to change (in ~/work); repeat for extra references (up to 4)")
    ap.add_argument("--no-expand", action="store_true", help="FLUX 3: use the prompt as typed (no caption expansion)")
    ap.add_argument("--name", default="", help="short name for the output files (default: from the prompt)")
    ap.add_argument("--model", metavar="NAME", help=f"image model: {', '.join(PROFILES['models'])} (default {PROFILES['default']})")
    ap.add_argument("--style", choices=list(PROFILES["styles"]), default=None, help="image: photo or art (default: per model)")
    ap.add_argument("--negative", default=None, help='what to avoid (default: per model and style; "" = nothing)')
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--size", default="1024x1024", help="image size WxH (multiples of 64)")
    ap.add_argument("--upscale", action="store_true", help="image: also save a fast 2x copy (4x-UltraSharp)")
    ap.add_argument("--hires", action="store_true", help="image: also save a detailed 2x copy (tiles redrawn, slower)")
    ap.add_argument("--pose", metavar="IMAGE", help="image: copy the body pose from this picture (in ~/work)")
    ap.add_argument("--pose-strength", type=float, default=None, help="how strictly to follow the pose, 0.3-1.0 (default per model)")
    ap.add_argument("--face", metavar="IMAGE", help="image: keep this face (in ~/work; one clear face)")
    ap.add_argument("--face-weight", type=float, default=0.65, help="how strongly to keep the face, 0.3-1.2 (high values distort it)")
    ap.add_argument("--checkpoint", metavar="NAME", help="image: any SDXL checkpoint file by (part of) its name")
    ap.add_argument("--fix-faces", action="store_true", help="image: redraw faces in more detail")
    ap.add_argument("--fix-hands", action="store_true", help="image: redraw hands in more detail")
    ap.add_argument("--frames", type=int, default=33, help="video length, 4n+1 (33 ~ 1.4 s, 121 ~ 5 s)")
    ap.add_argument("--timeout", type=int, default=None, help="seconds (default 900 image, 3600 video)")
    a = ap.parse_args()
    # limits for a 12 GB GPU (and against a runaway agent): larger jobs fail or block the GPU for hours
    if a.steps is not None and not 1 <= a.steps <= MAX_STEPS:
        sys.exit(f"--steps must be 1-{MAX_STEPS}")
    if a.timeout is not None and not 60 <= a.timeout <= MAX_TIMEOUT:
        sys.exit(f"--timeout must be 60-{MAX_TIMEOUT} s")
    if a.kind == "video" and not 5 <= a.frames <= MAX_FRAMES:
        sys.exit(f"--frames must be 5-{MAX_FRAMES}")
    os.makedirs(OUTPUT, exist_ok=True)
    if shutil.disk_usage(OUTPUT).free < MIN_FREE:
        sys.exit(f"less than {MIN_FREE >> 30} GB free in {OUTPUT}")
    seed = a.seed if a.seed is not None else random.randint(1, 2**31)
    prefix = f"{datetime.date.today():%Y-%m-%d}/{slug(a.name or ' '.join(a.prompt.split()[:6]))}"
    if a.kind in ("image", "edit"):
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
        prof = find_checkpoint(a.checkpoint) if a.checkpoint else profile(a.model)
        arch, style = prof["arch"], a.style or prof.get("style", "photo")
        allowed = {"sdxl": {"pose", "face", "fix_faces", "fix_hands", "hires"}, "anima": {"pose"}}.get(arch, set())
        used = [o for o in ("pose", "face", "fix_faces", "fix_hands", "hires") if getattr(a, o)]
        if [o for o in used if o not in allowed]:
            sys.exit(f"--model {prof['name']} ({arch}) does not support --{', --'.join(o.replace('_', '-') for o in used if o not in allowed)}"
                     " - those work with the SDXL models (realvis, juggernaut, sdxl-base)" + (" and --pose with anima" if arch != "anima" else ""))
        if a.kind == "edit" and arch != "flux3":
            sys.exit("edit needs a FLUX 3 model (--model flux3)")
        if a.image and a.kind != "edit":
            sys.exit("--image is for edit")
        pose_strength = a.pose_strength if a.pose_strength is not None else prof.get("pose_strength", 0.65)
        if not (0.3 <= pose_strength <= 1.0 and 0.3 <= a.face_weight <= 1.2):
            sys.exit("--pose-strength must be 0.3-1.0 and --face-weight 0.3-1.2")
        neg = a.negative if a.negative is not None else negative_for(prof, style)
        steps = a.steps or prof["steps"]
        log0 = lambda t: print(f"  {t}", flush=True)
        prompt = a.prompt
        if arch == "sdxl":
            uploads = [(n, check_input(f)) for n, f in (("30", a.pose), ("21", a.face)) if f]
            wf = sdxl(prompt, neg, seed, steps, w, h, a.upscale, prefix, pose="pending" if a.pose else None,
                      face="pending" if a.face else None, fix_faces=a.fix_faces, fix_hands=a.fix_hands, hires=a.hires,
                      pose_strength=pose_strength, face_weight=a.face_weight, prof=prof)
        elif arch == "anima":
            uploads = [("30", check_input(a.pose))] if a.pose else []
            wf = anima(prompt, neg, seed, steps, w, h, prefix, prof, pose="pending" if a.pose else None,
                       pose_strength=pose_strength, upscale=a.upscale)
        elif arch == "zimage":
            uploads, neg = [], ""                           # CFG 1: no negative prompt
            wf = zimage(prompt, seed, steps, w, h, prefix, prof, upscale=a.upscale)
        elif arch == "flux3":
            refs = [check_input(f) for f in a.image]
            if not a.no_expand:
                prompt = expand_prompt(a.prompt, "edit" if refs else "t2i", w, h, refs, log=log0)
            uploads, neg = [(f"2{i}0", f) for i, f in enumerate(refs)], ""
            wf = flux3(prompt, seed, steps, w, h, prefix, prof, refs=["pending"] * len(refs), upscale=a.upscale)
        else:
            sys.exit(f"--model {prof['name']}: architecture {arch} is not wired into ai gen")
        info = {"kind": a.kind, "prompt": a.prompt, "expanded_prompt": prompt if prompt != a.prompt else None,
                "negative": neg, "seed": seed, "model": prof["name"], "arch": arch,
                "model_file": prof["file"], "sampler": prof["sampler"], "scheduler": prof["scheduler"], "cfg": prof["cfg"],
                "steps": steps, "size": f"{w}x{h}", "style": style,
                "options": {k: getattr(a, k) for k in ("upscale", "hires", "fix_faces", "fix_hands")},
                "pose": a.pose and {"file": a.pose, "strength": pose_strength},
                "face": a.face and {"file": a.face, "weight": a.face_weight}, "images": a.image or None}
        limit = a.timeout or (1800 if a.hires or a.fix_hands or a.fix_faces or arch == "flux3" else 900)
    else:
        frames = max(5, (a.frames - 1) // 4 * 4 + 1)
        if a.pose or a.face or a.fix_faces or a.fix_hands or a.hires or a.checkpoint or a.model or a.image:
            sys.exit("--pose/--face/--fix-faces/--fix-hands/--hires/--checkpoint/--model/--image are for images")
        uploads = []
        neg = a.negative if a.negative is not None else "static, blurry, distorted, low quality, watermark"
        wf, limit = wan(a.prompt, neg, seed, a.steps or 20, frames, prefix), a.timeout or 3600
        info = {"kind": "video", "prompt": a.prompt, "negative": neg, "seed": seed, "model": "wan2.2-ti2v-5b",
                "steps": a.steps or 20, "frames": frames}
    log = lambda s: print(f"  {s}", flush=True)
    ok, files, err = generate(wf, limit, log=log, uploads=uploads)
    if not ok and not files and re.search(r"read_file_slice|HostBuffer|deserializ|Unable to load|out of memory", err or "", re.I):
        log(f"model loading failed ({err[:80]}) - retrying once")      # seen once right after the chat model unloaded
        ok, files, err = generate(wf, limit, log=log, uploads=uploads)
    if err:
        print(f"  error: {err}")
    rec = write_record(files, info) if ok else None
    print(f"OUTPUT (seed {seed})" if ok else "FAILED")
    for f in files:
        print(f)
    if rec:
        print(f"  record: {rec}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
