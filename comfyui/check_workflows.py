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


def check(wf, schema, label, unknown):
    """! @brief Problems found in one workflow; node classes not in the schema yet are collected in unknown. @return list of text """
    bad = []
    for nid, node in wf.items():
        c = node["class_type"]
        if c not in schema:
            unknown.add(c); continue
        spec = schema[c]
        if spec is None:
            bad.append(f"{label}: node {nid} {c} is not known to the pinned ComfyUI"); continue
        for k, v in node["inputs"].items():
            base = k.split(".")[0]                      # dynamic inputs: images.image0, resolution.megapixels
            if base not in spec:
                bad.append(f"{label}: {c}.{k} is not an input of {c}")
            elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                if v[0] not in wf:
                    bad.append(f"{label}: {c}.{k} links to missing node {v[0]}")
            elif k == base and spec[k]["values"] is not None and v not in spec[k]["values"]:
                bad.append(f"{label}: {c}.{k} = {v!r} is not one of {spec[k]['values'][:8]}")
        for k, d in spec.items():
            if d["part"] == "required" and k not in node["inputs"] and not any(i.startswith(k + ".") for i in node["inputs"]):
                bad.append(f"{label}: {c} is missing required input {k}")
    return bad


def main():
    """! @brief Builds all variants and checks them. @return 0 if clean """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    if not os.path.exists(path):
        print("schema.json not captured yet (run: ai gen schema) - workflow check skipped"); return 0
    schema, bad, n, unknown = json.load(open(path)), [], 0, set()
    for i, wf in enumerate(gen.all_workflows()):
        bad += check(wf, schema, f"variant {i} ({wf[next(iter(wf))]['class_type']}...)", unknown); n += 1
    if unknown:   # new nodes: recapture with "ai gen schema" after installing; until then they are not checked
        print(f"not in schema.json yet (recapture with: ai gen schema): {', '.join(sorted(unknown))}")
    for b in sorted(set(bad)):
        print(b)
    print(f"{n} workflow variants checked: {'OK' if not bad else f'{len(set(bad))} problem(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
