"""Z-Image pose control: the Fun ControlNet Union patch (nodes 30-36; the picture is uploaded to node 30)."""
from features.pose.dwpose import skeleton


def control(wf, model, pose, patch, strength, prefix, vae, width, height):
    """! @return the pose-following model link """
    wf.update({
        "30": {"class_type": "LoadImage", "inputs": {"image": pose}},
        "31": skeleton(["30", 0]),
        "32": {"class_type": "ModelPatchLoader", "inputs": {"name": patch}},
        "33": {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
            "model": model, "model_patch": ["32", 0], "vae": vae, "image": ["36", 0], "strength": strength}},
        "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["36", 0]}},
        # Native control otherwise center-crops a differently shaped guide.
        "36": {"class_type": "ImageScale", "inputs": {"image": ["31", 0], "upscale_method": "nearest-exact",
               "width": width, "height": height, "crop": "disabled"}},
    })
    return ["33", 0]
