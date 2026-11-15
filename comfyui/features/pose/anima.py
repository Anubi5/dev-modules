"""Anima pose control: the LLLite pose patch (nodes 30-35; the picture is uploaded to node 30)."""
from features.pose.dwpose import skeleton


def control(wf, model, pose, patch, strength, prefix):
    """! @return the pose-following model link """
    wf.update({
        "30": {"class_type": "LoadImage", "inputs": {"image": pose}},
        "31": skeleton(["30", 0]),
        "32": {"class_type": "ModelPatchLoader", "inputs": {"name": patch}},
        "33": {"class_type": "AnimaLLLiteApply", "inputs": {"model": model, "model_patch": ["32", 0], "image": ["31", 0],
               "strength": strength, "start_percent": 0.0, "end_percent": 1.0}},
        "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["31", 0]}},
    })
    return ["33", 0]
