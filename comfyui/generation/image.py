"""Image and editing architecture adapters; no execution or GPU ownership."""
from .models import PROFILES, profile

from features import references, refine
from features import upscale as upscaler
from features.face import sdxl as face_keep
from features.face_fix import sdxl as face_fix
from features.face_fix.sdxl import detailer  # noqa: F401  (kept importable)
from features.hand_fix import sdxl as hand_fix
from features.pose import anima as anima_pose, sdxl as sdxl_pose, zimage as zimage_pose
from features.pose.dwpose import skeleton as _dwpose  # noqa: F401  (kept importable)

FACE_DET, HAND_DET = face_fix.DETECTOR, hand_fix.DETECTOR
POSE_CN = sdxl_pose.CONTROLNET


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


def _loaders(prof, clip_type):
    """! @brief Model, text encoder and VAE loader nodes "1","2","3" for the newer (non-SDXL) architectures. """
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": prof["file"], "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": prof["text_encoder"], "type": clip_type, "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": prof["vae"]}}}


def _finish(wf, prof, seed, steps, model, pos, neg, latent, prefix, upscale, hires=False, fix_faces=False):
    """!
    @brief KSampler + decode + save (+ optional fast 2x copy, + optional refined 2x copy) shared by the newer architectures.
    @details hires: the image is enlarged 2x with the upscale model, then redrawn lightly by the same model (few steps,
             low denoise: profile keys refine_steps / refine_denoise). That adds real detail (skin, hair, fabric)
             without changing the face or the pose - a plain upscale only enlarges what is already there.
    """
    wf.update({
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": prof["cfg"], "sampler_name": prof["sampler"],
              "scheduler": prof["scheduler"], "denoise": 1.0, "model": model, "positive": pos, "negative": neg, "latent_image": latent}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["9", 0]}},
    })
    if fix_faces:
        wf["27"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_base", "images": ["9", 0]}}
        # The installed Z-Image crop sampler overprocesses skin. Reuse the
        # existing SDXL repair path with its own model, conditioning and VAE.
        wf["10"]["inputs"]["images"] = face_fix.with_sdxl_refiner(wf, ["9", 0], wf["4"]["inputs"]["text"], seed,
                                                                  profile(PROFILES.get("default_sdxl", "realvis")))
    if upscale:
        wf["11"] = upscaler.loader()
        upscaler.fast_2x(wf, "11", ["9", 0], prefix, "12", "13", "14")
    if hires:
        refine.latent_2x(wf, ["9", 0], model, pos, neg, ["3", 0], seed, prof, prefix)
    if fix_faces:
        if upscale: wf["12"]["inputs"]["image"] = ["41", 0]
        if hires: wf["21"]["inputs"]["image"] = ["41", 0]
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
    for i, link in enumerate(references.load(wf, refs)):
        rml[f"images.image{i}"] = link
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
        model = anima_pose.control(wf, model, pose, prof["pose_patch"], pose_strength, prefix)
    return _finish(wf, prof, seed, steps, model, ["4", 0], ["5", 0], ["6", 0], prefix, upscale)


def zimage(prompt, seed, steps, width, height, prefix, prof, upscale=False, hires=False, fix_faces=False, pose=None, pose_strength=1.0):
    """!
    @brief Z-Image Turbo text-to-image (8 steps, CFG 1, AuraFlow sampling shift 3 - official template).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "lumina2")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
               "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
               "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": prof.get("shift", 3.0)}}})
    model = ["7", 0]
    if pose:
        model = zimage_pose.control(wf, model, pose, prof["pose_patch"], pose_strength, prefix, ["3", 0], width, height)
    return _finish(wf, prof, seed, steps, model, ["4", 0], ["5", 0], ["6", 0], prefix, upscale, hires,
                   fix_faces=fix_faces)


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
    for i, link in enumerate(references.load(wf, refs)):
        enc[f"images.image_{i + 1}"] = link
    wf["4"] = {"class_type": "TextEncodeFibo2", "inputs": enc}
    return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["4", 1], ["4", 2], prefix, upscale)


POSE_PROMPT = ("wide full body photo of a person {d}, entire body from head to feet in frame, space around the figure, "
               "plain light grey studio background, even soft light")


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
