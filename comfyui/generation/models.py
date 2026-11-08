"""Declarative shipped model profiles and style defaults."""
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass


with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles.json")) as stream:
    PROFILES = json.load(stream)

from .registry import profiles as family_profiles
_families = family_profiles()
if set(_families) & set(PROFILES['models']): raise ValueError('Family package shadows an existing model profile')
PROFILES['models'].update(_families)


NEGATIVE = PROFILES["styles"]["photo"]["negative"]            # default style; kept as a name for the tests


def profile(name=None):
    """! @brief Settings for a model from profiles.json (default model when name is None). @return dict incl. "name" """
    name = name or PROFILES["default"]
    if name not in PROFILES["models"]:
        sys.exit(f"--model {name}: unknown. Models: {', '.join(PROFILES['models'])}")
    return dict(PROFILES["models"][name], name=name)


def negative_for(prof, style):
    """! @brief Default negative prompt for a model and style (the model's own, else the style's). @return text """
    return prof.get("negative", {}).get(style, PROFILES["styles"][style]["negative"])


def arch_base(arch):
    """! @brief The shipped profile an architecture's other files start from (default_sdxl for SDXL, else the first). """
    if arch == "sdxl":
        return profile(PROFILES.get("default_sdxl", "realvis"))
    name = next((n for n, p in PROFILES["models"].items() if p["arch"] == arch), None)
    if not name:
        sys.exit(f"architecture {arch!r} is not wired into ai gen. Known: {', '.join(sorted(archs()))}")
    return profile(name)


def archs():
    """! @brief Image architectures ai gen has workflows for (3D models run through ai gen 3d). """
    return {p["arch"] for p in PROFILES["models"].values() if p.get("kind") != "3d"}


@dataclass(frozen=True)
class ModelPackage:
    """One declarative model selection; settings are a detached per-request copy."""
    name: str
    kind: str
    architecture: str
    settings: dict


def package(name=None, kind="image"):
    """Resolve a package without imports, downloads, services or model loading.

    Image/3D use the existing catalogue; video has its own capability namespace
    so existing `ai gen models` output and image-model selection stay compatible.
    Architecture adapters consume these settings; metadata never names Python code.
    """
    if PROFILES.get("schema_version", 1) != 1:
        raise ValueError("Unsupported model-package schema version")
    defaults = {"image": "default", "3d": "default_3d", "video": "default_video"}
    if kind not in defaults:
        raise ValueError("Unknown generation capability: " + str(kind))
    name = name or PROFILES.get(defaults[kind])
    catalogue = PROFILES.get("video_models", {}) if kind == "video" else PROFILES["models"]
    if name not in catalogue:
        raise ValueError("Unknown " + kind + " model package: " + str(name))
    settings = deepcopy(catalogue[name])
    actual_kind = settings.get("kind", "image")
    if actual_kind != kind:
        raise ValueError("Model package capability mismatch: " + str(name))
    architecture = settings.get("arch")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("Model package needs an architecture: " + str(name))
    required = ["file"]
    if kind == "3d": required += ["workflow"]
    if kind == "video": required += ["text_encoder", "vae", "width", "height", "fps", "shift", "cfg", "sampler", "scheduler"]
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError("Incomplete model package " + str(name) + ": " + ", ".join(missing))
    for key in ("file", "workflow", "text_encoder", "vae", "sampler", "scheduler"):
        if key in settings and (not isinstance(settings[key], str) or not settings[key].strip()):
            raise ValueError("Invalid " + key + " in model package " + str(name))
    if kind == "video":
        import math
        for key in ("width", "height", "fps", "shift", "cfg"):
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid " + key + " in model package " + str(name))
        if any(not isinstance(settings[key], int) for key in ("width", "height")):
            raise ValueError("Video dimensions must be integers in model package " + str(name))
    return ModelPackage(name, kind, architecture, settings)
