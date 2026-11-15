"""--upscale: a fast 2x copy with the upscale model (4x, then halved)."""

MODEL = "4x-UltraSharp.safetensors"


def loader():
    """! @brief The upscale model loader node. """
    return {"class_type": "UpscaleModelLoader", "inputs": {"model_name": MODEL}}


def fast_2x(wf, loader_id, image, prefix, up, scale, save):
    """! @brief Adds the 2x copy of image ("<prefix>_upscaled") at node ids up, scale, save. @return wf """
    wf.update({
        up: {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": [loader_id, 0], "image": image}},
        scale: {"class_type": "ImageScaleBy", "inputs": {"image": [up, 0], "upscale_method": "lanczos", "scale_by": 0.5}},
        save: {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_upscaled", "images": [scale, 0]}},
    })
    return wf
