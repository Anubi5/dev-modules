"""Pure ComfyUI graph transformations."""
import json
import sys


def apply_recipe(recipe, values):
    """!
    @brief Fills a recipe's declared inputs and nothing else. Each input names the node inputs it sets
           ([node, input] or [node, input, offset] for seeds that differ per stage); values outside an input's
           declared range or choices are refused; an input with "drop" that is not given is unlinked there (a missing
           optional view), and prune() then removes what only it used. @param values  {input name: value}  @return workflow (a copy)
    """
    wf = json.loads(json.dumps(recipe["workflow"]))
    outputs = sinks(wf)                          # taken before any input is unlinked
    for name, spec in recipe["inputs"].items():
        v = values.get(name, spec.get("default"))
        if v is None:
            if "drop" in spec:                   # an optional input not given (a view): unlinked, its branch pruned below
                node, key = spec["drop"]
                wf[node]["inputs"].pop(key, None)
            continue
        if "choices" in spec and v not in spec["choices"]:
            sys.exit(f"{name} {v}: choose one of {', '.join(map(str, spec['choices']))}")
        if ("min" in spec and v < spec["min"]) or ("max" in spec and v > spec["max"]):
            sys.exit(f"{name} {v}: must be {spec.get('min')}-{spec.get('max')}")
        for node, key, *off in spec["set"]:
            wf[node]["inputs"][key] = v + off[0] if off else v
    return prune(wf, outputs)


def _link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def sinks(wf):
    """! @brief The nodes nothing reads from: a workflow's outputs (save and preview nodes). """
    return [k for k in wf if k not in {v[0] for n in wf.values() for v in n["inputs"].values() if _link(v)}]


def prune(wf, outputs=None):
    """!
    @brief Reduces a workflow to what its chosen settings use: a switch whose choice is fixed (a constant, or a
           Primitive node's value) is replaced by a direct link to the chosen input, then every node no output depends on
           is dropped. So a recipe holding several models needs, loads and records only the selected one. Outputs are
           the nodes nothing reads from in the full workflow (outputs: given when inputs were unlinked since); switches
           decided at run time stay. @return wf (changed)
    """
    outputs = sinks(wf) if outputs is None else outputs

    def choice(k):
        n = wf.get(k, {})
        if n.get("class_type") != "ComfySwitchNode":
            return None
        s = n["inputs"].get("switch")
        if _link(s):
            p = wf.get(s[0], {})
            s = p["inputs"].get("value") if p.get("class_type", "").startswith("Primitive") else None
        return None if not isinstance(s, bool) else n["inputs"].get("on_true" if s else "on_false")

    for n in wf.values():
        for key, v in n["inputs"].items():
            seen = set()
            while _link(v) and v[1] == 0 and v[0] not in seen and _link(choice(v[0])):
                seen.add(v[0]); v = choice(v[0])
            n["inputs"][key] = v
    keep, todo = set(), list(outputs)
    while todo:
        k = todo.pop()
        if k in keep or k not in wf:
            continue
        keep.add(k)
        todo += [v[0] for v in wf[k]["inputs"].values() if _link(v)]
    for k in set(wf) - keep:
        del wf[k]
    return wf


def apply_loras(wf, loras, arch):
    """!
    @brief Chains LoRA loaders after the model loader: everything that used the base model and text encoder now uses the
           LoRA'd ones (sampler, pose/face nodes, repairs, upscale). SDXL: checkpoint outputs "4"/0,1; the newer models:
           "1" (model) and "2" (text encoder). @param loras  [(file, strength)]  @return the workflow, changed in place
    """
    if not loras:
        return wf
    model, clip = (["4", 0], ["4", 1]) if arch == "sdxl" else (["1", 0], ["2", 0])
    base_model, base_clip = model, clip
    ids = []
    for i, (name, strength) in enumerate(loras):
        nid = f"6{i}"
        wf[nid] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                          "strength_model": strength, "strength_clip": strength}}
        model, clip = [nid, 0], [nid, 1]
        ids.append(nid)
    for nid, node in wf.items():
        if nid in ids:
            continue
        for k, v in node["inputs"].items():
            if v == base_model:
                node["inputs"][k] = model
            elif v == base_clip:
                node["inputs"][k] = clip
    return wf


def use_skeleton(wf):
    """!
    @brief The --pose picture is already a skeleton (from ai gen pose or another tool): skip DWPose and feed it
           straight to the pose control (and do not save it again). @return the workflow, changed in place
    """
    if "31" in wf and wf["31"]["class_type"] == "DWPreprocessor":
        del wf["31"]
        wf.pop("35", None)
        for node in wf.values():
            for k, v in node["inputs"].items():
                if v == ["31", 0]:
                    node["inputs"][k] = ["30", 0]
    return wf
