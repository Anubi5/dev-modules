"""--lora: LoRA loaders chained after the model loader (the files and strengths are chosen by the operation)."""


def apply_loras(wf, loras, arch):
    """!
    @brief Chains LoRA loaders after the model loader: everything that used the base model and text encoder now uses the
           LoRA'd ones (sampler, pose/face nodes, repairs, upscale). SDXL: checkpoint outputs "4"/0,1; the newer models:
           "1" (model) and "2" (text encoder). @param loras  [(file, strength)]  @return the workflow, changed in place
    """
    if not loras:
        return wf
    from generation import registry
    if not registry.has_implementation(arch):
        raise ValueError(f"LoRA support is not declared for architecture {arch}")
    spec = registry.implementation(arch).CAPABILITIES.get('lora')
    if not spec:
        raise ValueError(f"LoRA support is not declared for architecture {arch}")
    for node, allowed in spec['loaders'].items():
        if wf.get(node, {}).get('class_type') not in allowed:
            raise ValueError(f"LoRA loader is incompatible with {arch} at node {node}")
    model, clip = list(spec['model']), list(spec['clip'])
    base_model, base_clip = model, clip
    ids = []
    for i, (name, strength) in enumerate(loras):
        nid = f"6{i}"
        if nid in wf:
            raise ValueError(f"LoRA node would overwrite existing node {nid}")
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
