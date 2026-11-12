"""Image and editing architecture adapters; no execution or GPU ownership."""
from .models import PROFILES, profile

FACE_DET, HAND_DET = "bbox/face_yolov8m.pt", "bbox/hand_yolov8s.pt"


POSE_CN = "controlnet-union-sdxl-promax.safetensors"


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


def _finish(wf, prof, seed, steps, model, pos, neg, latent, prefix, upscale, hires=False, fix_faces=False, face_model=None):
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
        wf["40"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DET}}
        inputs = detailer(["9", 0], face_model or model, pos, neg, ["40", 0], seed, 0.35, prof)
        inputs.update(clip=["2", 0], vae=["3", 0], steps=steps, guide_size=1024, denoise=0.15)
        wf["41"] = {"class_type": "FaceDetailer", "inputs": inputs}
        wf["10"]["inputs"]["images"] = ["41", 0]
    if upscale:
        wf.update({
            "11": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.safetensors"}},
            "12": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["11", 0], "image": ["9", 0]}},
            "13": {"class_type": "ImageScaleBy", "inputs": {"image": ["12", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
            "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_upscaled", "images": ["13", 0]}},
        })
    if hires:
        wf.update({
            "20": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.safetensors"}},
            "21": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["20", 0], "image": ["9", 0]}},
            "22": {"class_type": "ImageScaleBy", "inputs": {"image": ["21", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
            "23": {"class_type": "VAEEncode", "inputs": {"pixels": ["22", 0], "vae": ["3", 0]}},
            "24": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": prof.get("refine_steps", 5), "cfg": prof["cfg"],
                   "sampler_name": prof["sampler"], "scheduler": prof["scheduler"], "denoise": prof.get("refine_denoise", 0.3),
                   "model": model, "positive": pos, "negative": neg, "latent_image": ["23", 0]}},
            "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["3", 0]}},
            "26": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_hires", "images": ["25", 0]}},
        })
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
        wf.update({
            "30": {"class_type": "LoadImage", "inputs": {"image": pose}},
            "31": _dwpose(["30", 0]),
            "32": {"class_type": "ModelPatchLoader", "inputs": {"name": prof["pose_patch"]}},
            "33": {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
                "model": model, "model_patch": ["32", 0], "vae": ["3", 0], "image": ["36", 0], "strength": pose_strength}},
            "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["36", 0]}},
            # Native control otherwise center-crops a differently shaped guide.
            "36": {"class_type": "ImageScale", "inputs": {"image": ["31", 0], "upscale_method": "nearest-exact",
                   "width": width, "height": height, "crop": "disabled"}},
        })
        model = ["33", 0]
    return _finish(wf, prof, seed, steps, model, ["4", 0], ["5", 0], ["6", 0], prefix, upscale, hires,
                   fix_faces=fix_faces, face_model=["7", 0])


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
