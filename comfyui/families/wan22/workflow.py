"""Reviewed Wan 2.2 video graph construction."""
from generation.models import package
ARCHITECTURES = frozenset({"wan22"})

def wan(prompt, negative, seed, steps, frames, prefix, model=None, image=None):
    """!
    @brief Wan 2.2 5B (GGUF Q8) text/image-to-video workflow, 640x352 at 24 fps, saved as animated WebP.
    @param prompt  What happens.  @param negative  What to avoid.  @param seed  Random seed.
    @param steps   Sampling steps.  @param frames  Length (4n+1, e.g. 33 = ~1.4 s, 121 = 5 s).
    @param prefix  Output prefix "<date>/<name>".
    @return Workflow in ComfyUI API format.
    """
    selected = package(model, kind="video")
    if selected.architecture not in ARCHITECTURES:
        raise ValueError("Wan video adapter needs architecture wan22, got " + selected.architecture)
    prof = selected.settings
    workflow = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": prof["file"]}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": prof["text_encoder"],
              "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": prof["vae"]}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"shift": prof["shift"], "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": prof["width"], "height": prof["height"],
              "length": frames, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": prof["cfg"], "sampler_name": prof["sampler"],
              "scheduler": prof["scheduler"], "denoise": 1.0, "model": ["4", 0], "positive": ["5", 0],
              "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["9", 0], "filename_prefix": prefix,
               "fps": prof["fps"], "lossless": False, "quality": 85, "method": "default"}},
    }
    if image is not None:
        workflow["11"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        workflow["7"]["inputs"]["start_image"] = ["11", 0]
    return workflow

def build(selected, prompt, negative, seed, steps, frames, prefix, image=None):
    return wan(prompt, negative, seed, steps, frames, prefix, model=selected.name, image=image)

CAPABILITIES = dict(timeout=3600, image_conditioning=True)
