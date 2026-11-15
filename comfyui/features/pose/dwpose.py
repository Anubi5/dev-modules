"""The DWPose skeleton every pose control reads, and --pose-skeleton (the picture already is one)."""


def skeleton(image, scale_stick="disable"):
    """! @brief DWPose skeleton node (body, hands, face) for an image link. """
    return {"class_type": "DWPreprocessor", "inputs": {"image": image, "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "enable", "resolution": 1024, "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384.onnx", "scale_stick_for_xinsr_cn": scale_stick}}


def use_skeleton(wf):
    """!
    @brief The --pose picture is already a skeleton (from ai gen pose or another tool): skip DWPose and feed it
           straight to the pose control (and do not save it again). @return the workflow, changed in place
    """
    if "31" in wf and wf["31"]["class_type"] == "DWPreprocessor":
        del wf["31"]
        wf.pop("35", None)
        for node in wf.values():
            for k, v in node["inputs"].items():
                if v == ["31", 0]:
                    node["inputs"][k] = ["30", 0]
    return wf
