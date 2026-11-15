"""!
@file video.py
@brief ai gen video: a short clip from a prompt with the configured video package.
"""
import sys

import runtime as rt
from generation.models import PROFILES
from generation.video import build


def validate(a):
    """! @brief Video's own limit, checked with the general ones before anything else. """
    if not 5 <= a.frames <= rt.MAX_FRAMES:
        sys.exit(f"--frames must be 5-{rt.MAX_FRAMES}")


def prepare(a, seed, prefix):
    """! @return (workflow, uploads, info, time limit, None) """
    frames = max(5, (a.frames - 1) // 4 * 4 + 1)
    if a.lora or a.pose_skeleton or a.pose or a.face or a.fix_faces or a.fix_hands or a.hires or a.checkpoint or a.image:
        sys.exit("--pose/--face/--fix-faces/--fix-hands/--hires/--checkpoint/--image are for images")
    uploads = []
    neg = a.negative if a.negative is not None else "static, blurry, distorted, low quality, watermark"
    wf, limit = build(a.prompt, neg, seed, a.steps or 20, frames, prefix, model=a.model), a.timeout or 3600
    info = {"kind": "video", "prompt": a.prompt, "negative": neg, "seed": seed, "model": a.model or PROFILES.get("default_video", "wan2.2-ti2v-5b"),
            "steps": a.steps or 20, "frames": frames}
    return wf, uploads, info, limit, None
