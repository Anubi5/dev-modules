"""Shared loader, sampling and output primitives; family composition stays with its owner."""
from features import refine
from features import upscale as upscaler


def _loaders(prof, clip_type):
    """! @brief Model, text encoder and VAE loader nodes "1","2","3" for the newer (non-SDXL) architectures. """
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": prof["file"], "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": prof["text_encoder"], "type": clip_type, "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": prof["vae"]}}}


def _finish(wf, prof, seed, steps, model, pos, neg, latent, prefix, upscale, hires=False, postprocess=None):
    """!
    @brief KSampler + decode + save (+ optional fast 2x copy, + optional refined 2x copy) shared by the newer architectures.
    @details hires: the image is enlarged 2x with the upscale model, then redrawn lightly by the same model (few steps,
             low denoise: profile keys refine_steps / refine_denoise). That adds real detail (skin, hair, fabric)
             without changing the face or the pose - a plain upscale only enlarges what is already there.
    """
    wf.update({
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": prof["cfg"], "sampler_name": prof["sampler"],
              "scheduler": prof["scheduler"], "denoise": 1.0, "model": model, "positive": pos, "negative": neg, "latent_image": latent}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["9", 0]}},
    })
    output = postprocess(wf) if postprocess else ['9', 0]
    wf['10']['inputs']['images'] = output
    if upscale:
        wf["11"] = upscaler.loader()
        upscaler.fast_2x(wf, "11", ["9", 0], prefix, "12", "13", "14")
    if hires:
        refine.latent_2x(wf, ["9", 0], model, pos, neg, ["3", 0], seed, prof, prefix)
    if postprocess:
        if upscale: wf["12"]["inputs"]["image"] = output
        if hires: wf["21"]["inputs"]["image"] = output
    return wf
