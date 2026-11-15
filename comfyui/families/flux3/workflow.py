"""Reviewed flux3 graph construction and feature composition."""
from features import references
from generation.shared import _loaders, _finish


def flux3(prompt, seed, steps, width, height, prefix, prof, refs=(), upscale=False, megapixels=1.0):
    """!
    @brief FLUX 3 Image Klein (distilled) text-to-image, or editing when reference images are given (BFL/Comfy template:
           CFG 1, references through ReferenceMultiLatent, the output takes the first reference's size).
    @param prompt  The full (expanded) caption.  @param refs  Uploaded image names (first = <ref_image_0>).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "flux3")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}})
    if not refs:
        wf["6"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["5", 0], ["6", 0], prefix, upscale)
    rml = {"positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0], "resolution": "resize", "resolution.megapixels": megapixels}
    for i, link in enumerate(references.load(wf, refs)):
        rml[f"images.image{i}"] = link
    wf["7"] = {"class_type": "ReferenceMultiLatent", "inputs": rml}
    edit = dict(prof, sampler=prof.get("edit_sampler", prof["sampler"]))
    return _finish(wf, edit, seed, steps, ["1", 0], ["7", 0], ["7", 1], ["7", 2], prefix, upscale)

def build(p, v, o):
    return flux3(v['prompt'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p,
        refs=['pending']*len(o.image), upscale=o.upscale)

PROMPT_TEMPLATES = {kind: 'families/flux3/prompts/' + kind + '.txt' for kind in ('t2i', 'edit')}

CAPABILITIES = dict(prompt_templates=PROMPT_TEMPLATES, implementation_files=list(PROMPT_TEMPLATES.values()), build=build, negative=False, edit=True, expand=True, timeout=1800)
