"""Reviewed moss graph construction and feature composition."""
from generation.shared import _loaders, _finish


def moss(prompt, negative, seed, steps, width, height, prefix, prof, upscale=False):
    """!
    @brief MOSS-Image text-to-image (OpenMOSS; Comfy template image_moss_image_t2i: 25 steps, CFG 2.5, er_sde, simple,
           Qwen3.5-9B text encoder, FLUX.2 small VAE). Good at legible text in images.
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "moss_image")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
               "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}})
    return _finish(wf, prof, seed, steps, ["1", 0], ["4", 0], ["5", 0], ["6", 0], prefix, upscale)

def build(p, v, o):
    return moss(v['prompt'], v['negative'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p, upscale=o.upscale)

CAPABILITIES = dict(lora={'model': ['1', 0], 'clip': ['2', 0], 'loaders': {'1': ['UNETLoader'], '2': ['CLIPLoader']}}, build=build, timeout=1800)
