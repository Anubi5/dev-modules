"""--fix-hands: detected hands redrawn by the SDXL model (Impact Pack FaceDetailer with the hand detector)."""
from features.face_fix.sdxl import detailer

DETECTOR = "bbox/hand_yolov8s.pt"


def on_sdxl(wf, image, model, pos, neg, seed, prof):
    """! @brief Nodes 42, 43. @return the repaired image link """
    wf["42"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": DETECTOR}}
    wf["43"] = {"class_type": "FaceDetailer", "inputs": detailer(image, model, pos, neg, ["42", 0], seed, 0.45, prof)}
    return ["43", 0]
