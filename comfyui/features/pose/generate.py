"""Pose operation composition: explicit reference extraction or the configured Z-Image pose source."""
from generation.models import profile
from families.zimage.workflow import zimage
from features.pose.dwpose import skeleton as _dwpose

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
