#!/usr/bin/env python3
"""!
@file check_workflows.py
@brief Checks every ai gen workflow variant against schema.json (node input definitions captured from the pinned ComfyUI
       with "ai gen schema"): node classes exist, required inputs are set, no unknown inputs, links point to existing
       nodes, choice values (samplers, schedulers, modes ...) are allowed. Runs without a GPU or ComfyUI (CI).
@return exit code 0 when all variants pass (or schema.json is missing: then it says so and passes).
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402


def check(wf, schema, label):
    """! @brief Problems found in one workflow. @return list of text """
    bad = []
    for nid, node in wf.items():
        c, spec = node["class_type"], schema.get(node["class_type"])
        if spec is None:
            bad.append(f"{label}: node {nid} {c} is not known to the pinned ComfyUI"); continue
        for k, v in node["inputs"].items():
            if k not in spec:
                bad.append(f"{label}: {c}.{k} is not an input of {c}")
            elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                if v[0] not in wf:
                    bad.append(f"{label}: {c}.{k} links to missing node {v[0]}")
            elif spec[k]["values"] is not None and v not in spec[k]["values"]:
                bad.append(f"{label}: {c}.{k} = {v!r} is not one of {spec[k]['values'][:8]}")
        for k, d in spec.items():
            if d["part"] == "required" and k not in node["inputs"]:
                bad.append(f"{label}: {c} is missing required input {k}")
    return bad


def main():
    """! @brief Builds all variants and checks them. @return 0 if clean """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    if not os.path.exists(path):
        print("schema.json not captured yet (run: ai gen schema) - workflow check skipped"); return 0
    schema, bad, n = json.load(open(path)), [], 0
    for model in gen.PROFILES["models"]:
        prof = gen.profile(model)
        for pose, face, ff, fh, hires, up in itertools.product([None, "p"], [None, "f"], *[[False, True]] * 4):
            wf = gen.sdxl("x", "y", 1, prof["steps"], 1024, 1024, up, "t", pose=pose, face=face, fix_faces=ff,
                          fix_hands=fh, hires=hires, prof=prof)
            bad += check(wf, schema, f"{model} pose={bool(pose)} face={bool(face)} fix={ff},{fh} hires={hires} up={up}"); n += 1
    bad += check(gen.wan("x", "y", 1, 20, 33, "t"), schema, "wan video"); n += 1
    for b in sorted(set(bad)):
        print(b)
    print(f"{n} workflow variants checked: {'OK' if not bad else f'{len(set(bad))} problem(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
