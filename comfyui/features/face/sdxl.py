"""--face: keep this face (IPAdapter FaceID Plus v2 on an SDXL model; the picture is uploaded to node 21)."""


def keep(wf, model, face, weight):
    """! @brief Nodes 20-22. @return the face-keeping model link """
    wf.update({
        "20": {"class_type": "IPAdapterUnifiedLoaderFaceID", "inputs": {"model": model, "preset": "FACEID PLUS V2",
               "lora_strength": 0.6, "provider": "CUDA"}},
        "21": {"class_type": "LoadImage", "inputs": {"image": face}},
        "22": {"class_type": "IPAdapterFaceID", "inputs": {"model": ["20", 0], "ipadapter": ["20", 1], "image": ["21", 0],
               "weight": weight, "weight_faceidv2": 1.0, "weight_type": "linear", "combine_embeds": "concat",
               "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
    })
    return ["22", 0]
