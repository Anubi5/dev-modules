"""!
@file pose.py
@brief ai gen pose: a pose picture (photo + skeleton) from a description, or the skeleton of a photo (--from).
"""
import sys

import runtime as rt
from generation.image import POSE_PROMPT, pose_wf
from operations import image


def prepare(a, seed, prefix):
    """! @return (workflow, uploads, info, time limit, None) """
    w, h = image.size(a, "896x1152")
    if a.pose or a.face or a.fix_faces or a.fix_hands or a.hires or a.checkpoint or a.model or a.image or a.lora:
        sys.exit("ai gen pose takes a description or --from PHOTO, plus --name/--seed/--size")
    uploads = [("30", rt.check_input(a.src))] if a.src else []
    wf = pose_wf(a.prompt, seed, w, h, prefix, src="pending" if a.src else None)
    info = {"kind": "pose", "prompt": a.prompt or None, "full_prompt": None if a.src else POSE_PROMPT.format(d=a.prompt),
            "from": a.src, "seed": None if a.src else seed, "model": None if a.src else "zimage", "size": f"{w}x{h}",
            "use": "ai gen image \"...\" --pose <the photo, or the _pose skeleton>"}
    limit = a.timeout or 900
    return wf, uploads, info, limit, None
