#!/usr/bin/env python3
"""!
@file detect.py
@brief Which kind of model a file is, by ComfyUI's own detection (comfy.model_detection, the code its loaders use):
       read from the file's header (tensor names and shapes), CPU only, no weights loaded, the GPU and the running
       ComfyUI untouched. Only when ComfyUI's detection needs a value (Z-Image reads one weight) are those few tensors read.
@details Run with ComfyUI's Python: detect.py FILE... -> JSON {file: class name or null}. Used by ai gen (gen.py), which
         maps the class to a workflow by detecting the files of its shipped profiles the same way.
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
COMFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "ComfyUI")
sys.path.insert(0, COMFY)
files, sys.argv = sys.argv[1:], [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()          # as ComfyUI's main.py: honour --cpu
import logging  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

logging.disable(logging.WARNING)
import comfy.model_detection as md  # noqa: E402

DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
      "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
      "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8), "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8)}


class Lazy(dict):
    """! @brief A state dict of shape-only (meta) tensors; real=True reads a tensor from disk when it is used. """

    def __init__(self, path, real=False):
        self.f, self.real = safe_open(path, "pt", device="cpu"), real
        super().__init__({k: None for k in self.f.keys()})

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        if self.real:
            return self.f.get_tensor(k)
        s = self.f.get_slice(k)
        return torch.empty(s.get_shape(), dtype=DT.get(s.get_dtype(), torch.float16), device="meta")

    def get(self, k, d=None):
        return self[k] if k in self else d

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]


def detect(path):
    """! @brief ComfyUI's model class for a .safetensors file, or None. """
    if not path.endswith(".safetensors"):
        return None
    meta = safe_open(path, "pt", device="cpu").metadata() or {}
    for real in (False, True):
        sd = Lazy(path, real)
        try:
            prefix = md.unet_prefix_from_state_dict(sd)
            if not any(k.startswith(prefix) for k in sd.keys()):      # as comfy.sd's diffusion-model loader
                prefix = ""
            cfg = md.model_config_from_unet(sd, prefix, metadata=meta)
            return type(cfg).__name__ if cfg else None
        except Exception:                                             # a value was needed: retry with real tensors
            continue
    return None


if __name__ == "__main__":
    out = {}
    for f in files:
        try:
            out[f] = detect(f)
        except Exception:
            out[f] = None
    print(json.dumps(out))
