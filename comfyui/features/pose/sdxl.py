"""SDXL pose control: the union ControlNet in openpose mode (nodes 30-35; the picture is uploaded to node 30)."""
from features.pose.dwpose import skeleton

CONTROLNET = "controlnet-union-sdxl-promax.safetensors"


def control(wf, pos, neg, pose, strength, prefix, vae):
    """! @return (positive, negative) conditioning links that follow the pose """
    wf.update({
        "30": {"class_type": "LoadImage", "inputs": {"image": pose}},
        "31": skeleton(["30", 0], "enable"),
        "32": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}},
        "33": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["32", 0], "type": "openpose"}},
        "34": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": pos, "negative": neg, "control_net": ["33", 0],
               "image": ["31", 0], "strength": strength, "start_percent": 0.0, "end_percent": 0.75, "vae": vae}},
        "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_pose", "images": ["31", 0]}},
    })
    return ["34", 0], ["34", 1]
