---
name: gpu-generate
description: Generate an image (SDXL; optionally in a given pose, with a given face kept, faces/hands repaired, detailed upscale) or a short video (Wan 2.2) on the local GPU with "ai gen"; it frees the GPU from the chat model first and gives it back afterwards.
---

# GPU generate

## When to Use
The user asks for a picture, illustration, concept art, texture, reference image or short video clip.

## Procedure
1. Write a concrete prompt: subject, setting, style, lighting, camera. Add a negative prompt only if needed (the default
   one already removes the plastic "AI" look). For photos, describe it like a photographer: "RAW photo", lens (35mm,
   50mm, 85mm), real light (window light, overcast, golden hour), "natural skin texture", "film grain"; keep colours
   plausible - words like "vibrant", "hyperdetailed", "masterpiece" make images look more artificial, not less.
2. Pick a short name for the result (it becomes the file name), e.g. `castle-concept`.
3. Run ONE command and wait for it to finish (the GPU is shared; the chat model is unloaded meanwhile):
   - image: `ai gen image "<prompt>" --name <name> [--size 1024x1024] [--seed N]` plus, when asked for:
     - `--pose <image>`: copy the body pose from a picture (photo, sketch, Blender render). `--pose-strength 0.3-1.0` (0.65).
     - `--face <image>`: keep the face from a picture with ONE clear face. `--face-weight 0.3-1.2` (0.65; above ~0.8 faces get gaunt and distorted).
     - `--fix-faces` / `--fix-hands`: redraw faces / hands in more detail (use for people, especially full-body shots).
     - `--hires`: also save a detailed 2x copy (slow); `--upscale`: a fast 2x copy.
     - `--checkpoint <name>`: another SDXL model, only when the user asks for one (default: RealVisXL, photoreal).
     Reference images must be inside `~/work` (usually `~/work/inputs/`) as .png/.jpg/.webp.
   - video: `ai gen video "<prompt>" --name <name> [--frames 33]`  (33 frames ~ 1.4 s, ~2-6 min; 121 ~ 5 s)
4. The last lines after `OUTPUT` are the file paths (in `~/work/outputs/comfy/<date>/`). Report them and the seed.
5. If the result belongs to a project, copy it into `~/work/projects/<project>/output/`.

## Rules
- Never run two `ai gen` commands at the same time; never call the ComfyUI API directly.
- If it prints `GPU busy` or `ComfyUI is busy`, wait and retry once. If it prints `FAILED`, report the error line - do not loop.
- Limits: sides 512-1536 px, up to 60 steps, up to 121 video frames. Don't chat-request other models while a job runs.
- `--face` only with faces of fictional characters or of people who agreed; never to put a real person into sexual or
  degrading images. If FaceID reports no face found, ask for a clearer front-facing face image.
- With `--pose`, a `<name>_pose_*.png` (the detected skeleton) is saved too - check it when the pose looks wrong.
- Sizes are multiples of 64; 1024x1024, 1152x896 and 896x1152 work best for SDXL.
