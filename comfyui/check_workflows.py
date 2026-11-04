#!/usr/bin/env python3
"""!
@file check_workflows.py
@brief Checks every ai gen workflow variant against schema.json (node input definitions captured from the pinned ComfyUI
       with "ai gen schema"): node classes exist, required inputs are set, no unknown inputs, links point to existing
       nodes, choice values (samplers, schedulers, modes ...) are allowed. Runs without a GPU or ComfyUI (CI).
@return exit code 0 when all variants and self-tests pass. A missing schema.json or a node class not in it is a
       warning locally and a failure in CI (CI set).
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


def self_tests():
    """!
    @brief Checks of ai gen's own logic that need no ComfyUI: an SDXL checkpoint without a profile gets the SDXL default
           settings (not the overall default, which may be another architecture); the preflight finds a file ComfyUI
           does not list. @return list of problems
    """
    import struct
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as d:
        head = json.dumps({"conditioner.embedders.1.model.x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        with open(os.path.join(d, "myOwnSDXL.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(head)) + head + b"\0\0")
        dirs, gen.CKPT_DIRS = gen.CKPT_DIRS, [d]
        try:
            p = gen.find_checkpoint("myown")
            if p["arch"] != "sdxl" or p["file"] != "myOwnSDXL.safetensors":
                bad.append(f"self-test: --checkpoint with an unprofiled SDXL file got arch {p['arch']} (needs sdxl)")
        finally:
            gen.CKPT_DIRS = dirs
    api = gen.api
    gen.api = lambda path, *a, **k: {"UNETLoader": {"input": {"required": {"unet_name": [["have.safetensors"]]}}}}
    try:
        wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing.safetensors"}}}
        ok = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "have.safetensors"}}}
        if gen.missing_files(wf) != ["UNETLoader.unet_name = missing.safetensors"] or gen.missing_files(ok):
            bad.append("self-test: the model-file preflight did not tell a missing file from a present one")
    finally:
        gen.api = api
    return bad


def main():
    """! @brief Builds all variants and checks them. @return 0 if clean """
    ci = bool(os.environ.get("CI"))     # GitHub Actions sets CI: there a missing schema or unknown node fails
    bad = self_tests()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    if not os.path.exists(path):
        print("schema.json not captured yet (run: ai gen schema)" + ("" if ci else " - workflow check skipped"))
        return 1 if ci or bad else 0
    schema, n, unknown = json.load(open(path)), 0, set()
    for i, wf in enumerate(gen.all_workflows()):
        bad += check(wf, schema, f"variant {i} ({wf[next(iter(wf))]['class_type']}...)", unknown); n += 1
    if unknown:   # new nodes: recapture with "ai gen schema" after installing; locally a warning, in CI a failure
        msg = f"not in schema.json yet (recapture with: ai gen schema): {', '.join(sorted(unknown))}"
        if ci:
            bad.append(msg)
        else:
            print("WARNING " + msg)
    for b in sorted(set(bad)):
        print(b)
    print(f"{n} workflow variants checked: {'OK' if not bad else f'{len(set(bad))} problem(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
