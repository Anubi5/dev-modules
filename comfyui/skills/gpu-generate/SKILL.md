---
name: gpu-generate
description: Generate or edit an image (SDXL photoreal with pose/face/repair, FLUX 3, Z-Image, Anima for anime/illustration) or a short video (Wan 2.2) on the local GPU with "ai gen"; it frees the GPU from the chat model first and gives it back afterwards.
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
     - Which model (leave `--model` out unless there is a reason; defaults were picked by a side-by-side test):
       plain photo/scene/product -> default (zimage); with --pose/--face/--fix-*/--hires -> juggernaut automatically
       (`--model realvis` for close-up portraits with --face); text in the image, posters, complex multi-part scenes
       -> `--model flux3`; anime/illustration -> `--model anima` (`--style art` for other artwork).
     - `--model <name>` (each brings its own tuned settings; `ai gen models` shows which are ready):
       realvis = photoreal portraits; the SDXL family (realvis, juggernaut, sdxl-base) is the only one with `--face`/`--fix-*`/`--hires`; juggernaut = natural
       documentary photos; sdxl-base = general/artistic; flux3 = follows long prompts, renders text, best composition
       (prompt is expanded automatically; optional - may not be installed); zimage = fast photoreal (8 steps);
       anima = anime/illustration (danbooru-style tags or plain text, `--pose` works); anima-turbo = fast anime drafts.
     - `--style art` for illustration, concept art, paintings, textures (the default `photo` style pushes away from art).
     - `--negative ""` means no negative prompt at all.
     Reference images must be inside `~/work` (usually `~/work/inputs/`) as .png/.jpg/.webp.
   - edit: `ai gen edit "<what to change>" --image <file> [--image <extra reference>] --name <name>` (FLUX 3): change
     an existing picture by instruction (recolour, replace, remove, restyle, put the person from image 1 into scene 2).
     Describe the change; everything else stays. To change the setting, say so explicitly
     ("change the background completely") - otherwise FLUX 3 keeps the original scene. The output is a new file - the original is never touched.
   - video: `ai gen video "<prompt>" --name <name> [--frames 33]`  (33 frames ~ 1.4 s, ~2-6 min; 121 ~ 5 s)
4. The lines after `OUTPUT` are the file paths (in `~/work/outputs/comfy/<date>/`); `record:` is a .json with every
   setting used. Report the files and the seed. With repair/upscale options, `<name>_base_*` is the untouched original -
   offer it if the repaired one looks worse.
   Before you report success, run `img-check <file>` and quote its line (size, contrast, main colours; BLANK = failed).
   If your vision tool works, also look at it and compare with the request (subject, pose, text, number of people,
   hands). Say honestly what is off instead of only listing the file.
5. If the result belongs to a project, copy it into `~/work/projects/<project>/output/`.

## Rules
- Never run two `ai gen` commands at the same time; never call the ComfyUI API directly.
- If it prints `GPU busy` or `ComfyUI is busy`, wait and retry once. If it prints `FAILED`, report the error line - do not loop.
- Limits: sides 512-1536 px, up to 60 steps, up to 121 video frames. Don't chat-request other models while a job runs.
- `--face` only with faces of fictional characters or of people who agreed; never to put a real person into sexual or
  degrading images. If FaceID reports no face found, ask for a clearer front-facing face image.
- With `--pose`, a `<name>_pose_*.png` (the detected skeleton) is saved too - check it when the pose looks wrong.
- Same character in several poses: follow the recipe in the comfyui module README (reference portrait, full-body pose
  sources with "entire body from head to feet in frame", identical character text + same seed; sitting, lying and
  crossed limbs go through the FLUX 3 two-reference route described there).
- Sizes are multiples of 64; 1024x1024, 1152x896 and 896x1152 work best for SDXL.
