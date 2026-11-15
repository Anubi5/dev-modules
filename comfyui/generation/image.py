"""Compatibility imports for the established graph API; implementations belong to their families."""
from families.sdxl.workflow import sdxl
from families.anima.workflow import anima
from families.zimage.workflow import zimage
from families.flux3.workflow import flux3
from families.fibo2.workflow import fibo2
from families.moss.workflow import moss
from generation.shared import _loaders, _finish
from features.pose.generate import POSE_PROMPT, pose_wf
from features.pose.dwpose import skeleton as _dwpose
from features.face_fix.sdxl import DETECTOR as FACE_DET, detailer
from features.hand_fix.sdxl import DETECTOR as HAND_DET
from features.pose.sdxl import CONTROLNET as POSE_CN
