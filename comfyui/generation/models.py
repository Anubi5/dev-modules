"""Declarative shipped model profiles and style defaults."""
import json
import os
import sys


with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles.json")) as stream:
    PROFILES = json.load(stream)


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
