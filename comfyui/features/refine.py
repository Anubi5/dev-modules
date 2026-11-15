"""--hires: a detailed 2x copy - the image enlarged, then redrawn lightly by the same model."""
from features import upscale


def tiled_sdxl(wf, image, model, pos, neg, vae, loader_id, seed, prof, prefix):
    """! @brief SDXL: Ultimate SD Upscale redraws 1024 px tiles (nodes 50, 51; upscale model loader_id). @return wf """
    S, K, C = prof["sampler"], prof["scheduler"], prof["cfg"]
    wf.update({
        "50": {"class_type": "UltimateSDUpscale", "inputs": {"image": image, "model": model, "positive": pos, "negative": neg,
               "vae": vae, "upscale_by": 2.0, "seed": seed, "steps": 20, "cfg": C, "sampler_name": S,
               "scheduler": K, "denoise": 0.2, "upscale_model": [loader_id, 0], "mode_type": "Linear", "tile_width": 1024,
               "tile_height": 1024, "mask_blur": 12, "tile_padding": 48, "seam_fix_mode": "None", "seam_fix_denoise": 1.0,
               "seam_fix_width": 64, "seam_fix_mask_blur": 8, "seam_fix_padding": 16, "force_uniform_tiles": True,
               "tiled_decode": False, "batch_size": 1}},
        "51": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_hires", "images": ["50", 0]}},
    })
    return wf


def latent_2x(wf, image, model, pos, neg, vae, seed, prof, prefix):
    """!
    @brief The newer architectures: enlarge 2x with the upscale model, then a short low-denoise pass of the same model
           (profile keys refine_steps / refine_denoise) adds real detail without changing face or pose (nodes 20-26).
    @return wf
    """
    wf.update({
        "20": upscale.loader(),
        "21": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["20", 0], "image": image}},
        "22": {"class_type": "ImageScaleBy", "inputs": {"image": ["21", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
        "23": {"class_type": "VAEEncode", "inputs": {"pixels": ["22", 0], "vae": vae}},
        "24": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": prof.get("refine_steps", 5), "cfg": prof["cfg"],
               "sampler_name": prof["sampler"], "scheduler": prof["scheduler"], "denoise": prof.get("refine_denoise", 0.3),
               "model": model, "positive": pos, "negative": neg, "latent_image": ["23", 0]}},
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": vae}},
        "26": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix + "_hires", "images": ["25", 0]}},
    })
    return wf
