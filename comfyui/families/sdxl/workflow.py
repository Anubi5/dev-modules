"""Reviewed sdxl graph construction and feature composition."""
from generation.models import PROFILES, profile
from features import refine
from features import upscale as upscaler
from features.face import sdxl as face_keep
from features.face_fix import sdxl as face_fix
from features.hand_fix import sdxl as hand_fix
from features.pose import sdxl as sdxl_pose


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
        model = face_keep.keep(wf, model, face, face_weight)
    cpos, cneg = pos, neg
    if pose:
        cpos, cneg = sdxl_pose.control(wf, pos, neg, pose, pose_strength, prefix, ["4", 2])
    wf.update({
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": C, "sampler_name": S,
              "scheduler": K, "denoise": 1.0, "model": model, "positive": cpos, "negative": cneg, "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    })
    image = ["8", 0]
    if save_base if save_base is not None else (fix_faces or fix_hands or hires):   # keep the original next to the repaired one
        wf["14"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_base", "images": image}}
    if fix_faces:   # repairs use the plain prompts (no pose steering on the crops) and the face-keeping model
        image = face_fix.on_sdxl(wf, image, model, pos, neg, seed, prof)
    if fix_hands:
        image = hand_fix.on_sdxl(wf, image, model, pos, neg, seed, prof)
    wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": image}}
    if upscale or hires:
        wf["10"] = upscaler.loader()
    if upscale:
        upscaler.fast_2x(wf, "10", image, prefix, "11", "12", "13")
    if hires:
        refine.tiled_sdxl(wf, image, model, pos, neg, ["4", 2], "10", seed, prof, prefix)
    return wf

def build(p, v, o):
    return sdxl(v['prompt'], v['negative'], v['seed'], v['steps'], v['width'], v['height'], o.upscale,
        v['prefix'], pose='pending' if o.pose else None, face='pending' if o.face else None,
        fix_faces=o.fix_faces, fix_hands=o.fix_hands, hires=o.hires, pose_strength=v['pose_strength'],
        face_weight=o.face_weight, prof=p)

CAPABILITIES = dict(lora={'model': ['4', 0], 'clip': ['4', 1], 'loaders': {'4': ['CheckpointLoaderSimple']}}, build=build, features=['pose','face','fix_faces','fix_hands','hires'], uploads={'pose':'30','face':'21'})
