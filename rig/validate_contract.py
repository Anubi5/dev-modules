"""Portable contract checks shared by the CLI, Blender validator and self-tests."""
import json
from pathlib import Path


class ContractError(ValueError):
    pass


def load_contract(path=None):
    contract = json.loads(Path(path or Path(__file__).with_name("contract.json")).read_text())
    bones = contract["required_bones"]
    if contract["schema_version"] != 1 or not bones or len(bones) != len(set(bones)):
        raise ContractError("Unknown schema or duplicate required bones")
    if not {"root", "pelvis", "head"} <= set(bones):
        raise ContractError("Missing root, pelvis or head contract")
    if contract["weights"]["max_influences"] != 4 or contract["units"]["metres_per_scene_unit"] != .01:
        raise ContractError("Humanoid v1 requires centimetres and at most four influences")
    if contract["mesh"]["default_triangle_budget"] > contract["mesh"]["max_triangles"]:
        raise ContractError("Default budget exceeds maximum")
    return contract


def validate_summary(summary, contract):
    """Validate measured facts; never infer a pass from file existence alone."""
    errors = []
    bones = summary.get("bones", [])
    missing = set(contract["required_bones"]) - set(bones)
    if missing:
        errors.append("missing bones: " + ", ".join(sorted(missing)))
    if len(bones) != len(set(bones)):
        errors.append("duplicate bone names")
    if summary.get("root_parent") is not None or summary.get("pelvis_parent") != "root" or summary.get("root_deforms", True):
        errors.append("root/pelvis hierarchy is invalid")
    if summary.get("max_influences", 999) > contract["weights"]["max_influences"]:
        errors.append("too many influences")
    if summary.get("unweighted_vertices", 1) != 0:
        errors.append("unweighted character vertices")
    if summary.get("max_weight_sum_error", 1) > contract["weights"]["sum_tolerance"]:
        errors.append("weights are not normalized")
    if summary.get("triangles", 10**12) > contract["mesh"]["max_triangles"]:
        errors.append("triangle budget exceeded")
    requested = summary.get("requested_height_m", contract["units"]["default_height_m"])
    if abs(summary.get("height_m", 0) - requested) > contract["units"]["height_tolerance_m"]:
        errors.append("height is wrong")
    if not summary.get("uv_layers") or not summary.get("materials"):
        errors.append("UVs/materials missing")
    return errors
