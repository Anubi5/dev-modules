"""--fix-faces: detected faces redrawn in more detail by an SDXL model (Impact Pack FaceDetailer)."""

DETECTOR = "bbox/face_yolov8m.pt"


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


def on_sdxl(wf, image, model, pos, neg, seed, prof):
    """! @brief An SDXL graph repairs its own faces (nodes 40, 41). @return the repaired image link """
    wf["40"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": DETECTOR}}
    wf["41"] = {"class_type": "FaceDetailer", "inputs": detailer(image, model, pos, neg, ["40", 0], seed, 0.35, prof)}
    return ["41", 0]


def with_sdxl_refiner(wf, image, prompt, seed, refiner):
    """!
    @brief Another architecture's faces repaired by an SDXL refiner model with its own conditioning and VAE (nodes 40,
           41, 44-46); the newer models' own crop sampler overprocesses skin. @return the repaired image link
    """
    if refiner["arch"] != "sdxl":
        raise ValueError("Z-Image face repair needs an SDXL default_sdxl profile")
    wf["40"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": DETECTOR}}
    wf["44"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": refiner["file"]}}
    wf["45"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["44", 1], "text": prompt}}
    wf["46"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["44", 1], "text": refiner["negative"]["photo"]}}
    inputs = detailer(image, ["44", 0], ["45", 0], ["46", 0], ["40", 0], seed, 0.2, refiner)
    inputs.update(clip=["44", 1], vae=["44", 2])
    wf["41"] = {"class_type": "FaceDetailer", "inputs": inputs}
    return ["41", 0]
