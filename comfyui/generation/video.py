"""Video workflow construction; no execution or GPU ownership."""


def wan(prompt, negative, seed, steps, frames, prefix):
    """!
    @brief Wan 2.2 5B (GGUF Q8) text-to-video workflow, 640x352 at 24 fps, saved as animated WebP.
    @param prompt  What happens.  @param negative  What to avoid.  @param seed  Random seed.
    @param steps   Sampling steps.  @param frames  Length (4n+1, e.g. 33 = ~1.4 s, 121 = 5 s).
    @param prefix  Output prefix "<date>/<name>".
    @return Workflow in ComfyUI API format.
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-TI2V-5B-Q8_0.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
              "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 640, "height": 352,
              "length": frames, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 5.0, "sampler_name": "uni_pc",
              "scheduler": "simple", "denoise": 1.0, "model": ["4", 0], "positive": ["5", 0],
              "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["9", 0], "filename_prefix": prefix,
               "fps": 24.0, "lossless": False, "quality": 85, "method": "default"}},
    }
