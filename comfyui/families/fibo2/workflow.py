"""Reviewed fibo2 graph construction and feature composition."""
from features import references
from generation.shared import _loaders, _finish


def fibo2(prompt, seed, steps, width, height, prefix, prof, refs=(), upscale=False):
    """!
    @brief Fibo 2.0 (Bria) text-to-image, or editing with up to 4 reference images (Comfy template
           image_fibo2_edit_turbo_pe: TextEncodeFibo2 makes both conditionings and the latent; empty negative = the
           null caption it was trained with). With references the result size is Fibo's trained bucket for the
           first image (about 1 MP); without, width x height.
    @param refs  Uploaded image names (<image_1>, <image_2> ... in the instruction).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "fibo2")
    enc = {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "vae": ["3", 0],
           "width": 0 if refs else width, "height": 0 if refs else height, "batch_size": 1}
    for i, link in enumerate(references.load(wf, refs)):
        enc[f"images.image_{i + 1}"] = link
    wf["4"] = {"class_type": "TextEncodeFibo2", "inputs": enc}
    return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["4", 1], ["4", 2], prefix, upscale)

def build(p, v, o):
    return fibo2(v['prompt'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p,
        refs=['pending']*len(o.image), upscale=o.upscale)

CAPABILITIES = dict(lora={'model': ['1', 0], 'clip': ['2', 0], 'loaders': {'1': ['UNETLoader'], '2': ['CLIPLoader']}}, fixed_cfg=True, build=build, negative=False, edit=True, timeout=1800)
