# comfyui (image and video generation)
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) with SDXL (photoreal RealVisXL V5.0 for `ai gen`, DPM++ SDE Karras, CFG 4.5) + 4x-UltraSharp and Wan 2.2 5B video (GGUF).
Page: http://127.0.0.1:8188 (`dev comfy`). Agents use `ai gen image|video "prompt"` (skill `gpu-generate`):
it unloads the chat model through the model-serving contract, runs one fixed workflow, frees the GPU.

- Custom nodes: `nodes.txt` (each pinned to a commit) · model files: `models.txt` - add a line, push, `dev mod update comfyui`.
- Pinned on purpose: ComfyUI itself (`COMFY_REF` in `install.sh`), every node (commit in `nodes.txt`) and every Python
  package (`app/constraints.txt`, written after the first install). Upgrade deliberately: raise the commits, or
  `COMFY_UPGRADE=1 dev mod update comfyui` for fresh Python packages. ComfyUI-Manager is not installed.
- Pickle-based model files (`.pt`, `.pth`, `.bin`) can run code when loaded: their hash is recorded on first fetch and a
  file that changes later is moved aside (`.quarantined`) instead of being used.
- Model files live only in D:\AI\models\comfy (read-only inside). Missing ones download straight to D: - that needs
  D: writable, which only `dev mod add|update comfyui` from Windows turns on (for the length of the run).
- `models.txt` takes Hugging Face lines and direct-download lines (`<https URL> <file> <folder> sha256:<hex>`, e.g. Juggernaut
  XL Ragnarok from CivitAI); a direct download is kept only if its SHA-256 matches. Models and their tuned settings live in
  `profiles.json` (`ai gen image ... --model realvis|juggernaut|sdxl-base`, `--style photo|art`); `--checkpoint <file>` uses
  any other SDXL file (checked to be SDXL). Every result gets a `.json` record (prompt, seed, model, settings, time, memory);
  with repair/upscale options the untouched original is kept as `<name>_base_*`.
- `ai gen schema` saves ComfyUI's node definitions (`~/work/outputs/comfy/schema.json`); committed as `schema.json`, the
  GitHub check validates every workflow variant against it (`check_workflows.py`) without a GPU.
- Your saved workflows and settings: `~/work/comfyui/` (kept on removal). Results: `~/work/outputs/comfy/<date>/`.
- `test.sh --full` = SDXL image + upscale, a Wan video, a pose + face + face/hand repair + tiled-upscale job, the newer
  models, and `ai gen pose` (photo + skeleton) with an image made from that skeleton, all through `ai gen`.

## Image models (`profiles.json`, `ai gen models` shows which are ready)
| `--model` | For | Notes |
|---|---|---|
| `zimage` (default) | photos, scenes, products - best realism in the test, ~25 s | Apache 2.0 |
| `realvis` | close-up portraits, best with `--face` | SDXL: the only family with `--pose --face --fix-faces --fix-hands --hires` |
| `juggernaut` | natural photos, products; the default when SDXL-only tools are used | SDXL, same tools |
| `sdxl-base` | general / artistic | SDXL, same tools |
| `flux3` | long prompts, text in images, layout; `ai gen edit` | optional (gated download, see models.txt); caption expanded by the local chat model with BFL's instructions (`prompts/`) |
| `anima`, `anima-turbo` | anime / illustration | `--pose` via Anima LLLite; non-commercial weights |
| `moss` | posters, signs, infographics: legible text, 1-2 MP | MOSS-Image SFT, Apache 2.0; INT8 + 4-bit text encoder (Comfy template settings). New 2 Nov 2026 - not yet in the side-by-side test |
| `fibo2` | `ai gen edit --model fibo2`: edits with up to 4 references, 4 steps; also text-to-image | Fibo 2.0 Edit Turbo, **CC BY-NC 4.0 (non-commercial)**; about 1 MP. Describe the final state of what changes - writing "keep X unchanged" makes it redraw the whole frame. New 2 Nov 2026 - test against the FLUX 3 pose route |

Not added yet: **Kandinsky 6.0** (open video with audio). It runs only through its own extension (registry packages
`kandinsky6`, `kandinsky6-sr`) with a "download models" button, and needs two large text encoders (Qwen3.5-9B bf16 and
Qwen2.5-VL-7B). Native ComfyUI support is under review (ComfyUI PR #16825); we add it once that is in a ComfyUI
release, so it uses the normal pinned-commit and checksum route like every other model here.

Settings come from each model's official template or card; tune them in `profiles.json`.

## Recipe: one character in several poses (tested 1 Nov 2026)
Best current route: SDXL (`realvis`) with `--face` + `--pose` + `--fix-faces --fix-hands` + `--hires`, about 3 min per image;
FLUX 3 with two references for poses SDXL gets wrong (see the end of this section).
1. **Character reference**: one clear, front-facing head-and-shoulders portrait of the character (`--model realvis`).
2. **Pose sources**: any person in each pose: `ai gen pose "<pose>"` makes a plain full-body photo plus its skeleton
   (`<name>_pose_...png`); either works with `--pose`. Only the skeleton is used, never the look.
3. **Final images**: the *same* character description (exact clothes, hair, eyes), the same `--seed`, plus
   `--face <reference> --pose <source> --fix-faces --fix-hands --hires`.
```powershell
$o = "/home/dev/work/outputs/comfy/$(Get-Date -f yyyy-MM-dd)"
$char = "RAW photo of a woman in her early 30s, shoulder-length wavy auburn hair, green eyes, light freckles, fitted olive-green bomber jacket over a white t-shirt, dark blue jeans, white sneakers, natural skin texture, 50mm, soft daylight"
dev ai gen image "$char, head and shoulders portrait, facing the camera, plain light grey background" --model realvis --size 896x1152 --seed 100 --name charref
$poses = "standing with hands on hips", "walking towards the camera", "jumping with both arms raised"
$i = 1; foreach ($p in $poses) {
  dev ai gen pose "$p" --seed 200 --name "posesrc$i"
  dev ai gen image "$char, full body, head to feet in frame, $p, city park background" --model realvis --face "$o/charref_00001_.png" --pose "$o/posesrc${i}_00001_.png" --fix-faces --fix-hands --hires --seed 300 --size 896x1152 --name "char-pose$i"
  $i++ }
dev share
```
What we learned:
- **The skeleton decides the framing.** If the pose source has no head (or feet) in frame, the result has none either.
  Always ask for "entire body from head to feet in frame" and check `<name>_pose_*.png` first when a result looks wrong.
- **Full-body pose sources fix most anatomy problems.** With a head-to-feet source, crouching (tying a shoelace) came
  out clean: correct hands, two sneakers. The earlier cropped source had given a third sneaker and a melted hand.
- **Sitting cross-legged fails on SDXL** - upside-down figure at pose strength 0.65 *and* 0.85, although the skeleton
  was correct. Overlapping legs are too ambiguous for the pose ControlNet. The FLUX 3 route below got the same pose
  right on the first try (clean hands, two feet, same outfit) - use it for lying, sitting cross-legged, tangled limbs.
- **Clothes stay consistent only through the text**: FaceID keeps the face, the description keeps the outfit - repeat
  it word for word. Same seed for all images keeps lighting and style close.
- **Small faces** (full body at 896x1152) need `--fix-faces`; likeness weak -> `--face-weight 0.75` (above ~0.8 faces
  get gaunt); pose too stiff -> `--pose-strength 0.55`.
- Hands are much better with `--fix-hands` and a full-body source, but not guaranteed in complex poses - keep the
  `_base_` original to compare, and re-run with another `--seed` if a hand is wrong.
- Colours of small garment parts can drift (a black instead of white t-shirt under a closed jacket): name the colour
  twice or add it to `--negative` (e.g. "black shirt").
- **FLUX 3 route** (hard poses, very consistent outfits): give the character and the pose source as two references:
  `ai gen edit "Create a photograph of the woman from the first image (same face, hair and outfit), in exactly the pose
  of the person in the second image: <pose>. Full body, head to feet in frame, <background>; change the background
  completely." --image <reference> --image <pose source>` - ~1.5 min, no FaceID or hand repair needed in the test.
  The face likeness comes from the reference image alone, so it is a bit looser than SDXL with `--face`.

## Adding a model, LoRA, checkpoint or node
Pick the lightest step that does what you need.

**1. Just try it (no repo change).** Put the file into the matching folder under `D:\AI\models\comfy\` and press `r`
in the ComfyUI page (or restart it: `dev ai stop comfyui`, `dev comfy`):

| File | Folder |
|---|---|
| SDXL checkpoint (one `.safetensors`, ~7 GB) | `checkpoints\` |
| LoRA | `loras\` |
| Diffusion model / text encoder / VAE of newer models (FLUX, Z-Image, Anima, Wan) | `diffusion_models\`, `text_encoders\`, `vae\` |
| Upscaler | `upscale_models\` |
| ControlNet / Anima LLLite patch | `controlnet\` / `model_patches\` |

It appears in the matching loader node at once. An SDXL checkpoint also works with ai gen right away:
`ai gen image "..." --checkpoint <part of its name>` (checked to be SDXL). LoRAs can only be used in the ComfyUI page
for now - ai gen has no `--lora` yet (on the to-do list). Prefer `.safetensors`; a `.pt`/`.ckpt` file can run code
when loaded.

**2. Keep it, with a checksum (fetched on every PC).** Add a line to `models.txt`, push, `dev mod update comfyui`:
```
<huggingface repo> <path in the repo> <folder> sha256:<hex>                      # Hugging Face
<https download URL> <file name> <folder> sha256:<hex>                           # e.g. CivitAI
... optional                                                                      # gated / manual download: missing is not an error
```
The SHA-256 is on the file's Hugging Face page ("SHA256") or CivitAI file details. A file whose checksum does not
match is refused (moved aside as `.quarantined`).

**3. Make it an ai gen model with its own settings.** Add an entry to `profiles.json` (copy one of the same `arch`:
`sdxl`, `flux3`, `zimage`, `anima`) with the file names and the sampler/scheduler/CFG/steps from the model's card,
push, `dev mod update comfyui`. `ai gen models` shows it, `ai gen image ... --model <name>` uses it, and the CI checks
its workflows. Record why in the entry's `source`.

**4. A new architecture or a custom node** is code: a workflow builder in `gen.py` (copy one from the model's official
ComfyUI template), its node pin in `nodes.txt` (`<git url> <commit>` - review the repo first, nodes run code), a job in
`comfy_test.py`, then `ai gen schema` + `dev share outputs/comfy/schema.json` so CI checks it. This reopens the frozen
module - only for a real need (see below).

## Image module 1.0 (frozen 1 Nov 2026) - tested on the RTX 4080 Laptop (12 GB)
| Job | Time | Peak GPU |
|---|---|---|
| SDXL image 1024² (realvis / juggernaut) | 18-31 s | 8-10.5 GB |
| SDXL + pose + face + face/hand repair + detailed 2x | ~190 s | 11.2 GB |
| Z-Image Turbo 1024² | 25-28 s | 10.5-10.8 GB |
| Anima (+ pose) 896x1152 | 33-37 s | 6.4-7 GB |
| FLUX 3 text-to-image (+ ~15 s caption expansion) | ~50 s | 11.0-11.4 GB |
| FLUX 3 edit with one reference | ~75 s | 11.1-11.6 GB |
| Wan 2.2 video, 33 frames | 75-86 s | 11.4-11.6 GB |

Known limits: sides 512-1536 px; FLUX 3 invents shapes on product shots (use zimage/juggernaut); FLUX 3 edits keep the
scene unless told to change it; jobs started in the ComfyUI page or chat requests during a job are not coordinated with
`ai gen` (run `ai free` first, don't chat during big jobs); a first FLUX 3 load right after the chat model can fail once
(`ai gen` retries automatically).

Frozen means: no new image models unless a needed task cannot be done, a model breaks, or a clearly better one is
measured on this machine. Settings changes go into `profiles.json` with the reason.

## From the command line (and for agents: skill `gpu-generate`)
```
ai gen image "a knight in a cafe" --name knight --pose ~/work/inputs/pose.png --face ~/work/inputs/face.png \
    --fix-faces --fix-hands --hires          # optional: --pose-strength 0.65 --face-weight 0.8 --size 896x1152
```
Same GPU hand-off as always (chat model unloaded, one job at a time). Reference images must be inside `~/work`.
Saves `<name>_*.png`, plus `<name>_pose_*.png` (the detected skeleton) and `<name>_hires_*.png` (detailed 2x).

## Control, faces, repair, upscaling (in the ComfyUI page)
| Want | Nodes (double-click the canvas, type the name) | Model files (models.txt) |
|---|---|---|
| Pose from a photo, skeleton or Blender render | `DWPose Estimator` (bbox `yolox_l.onnx`, pose `dw-ll_ucoco_384.onnx` - already installed) → `Apply ControlNet` (strength ~0.65, end ~0.75) | `controlnet-union-sdxl-promax` (one ControlNet for pose, depth, canny, lineart) |
| Depth / layout | `Depth Anything` or `MiDaS` preprocessor → same ControlNet | same |
| Same face across images | `IPAdapter FaceID` (weight ~0.65; higher distorts the face) with a face reference; lighter: `IPAdapter Advanced` + `ip-adapter-plus-face_sdxl` | FaceID Plus v2 + its LoRA, plus-face, CLIP-ViT-H |
| Fix faces / hands | `FaceDetailer` with `UltralyticsDetectorProvider` (`bbox/face_yolov8m.pt`, `bbox/hand_yolov8s.pt`), denoise ~0.3 | face/hand detectors |
| Big sharp output | `Ultimate SD Upscale` (upscale 2x, tile 768/1024, denoise ~0.2) with 4x-UltraSharp | already there |

- Already installed, so they work offline: the DWPose files (`helpers.txt`) and InsightFace's face models. Other
  preprocessors (depth, canny ...) still fetch their helper file on first use - internet on that once.
- 12 GB GPU: SDXL + ControlNet + IPAdapter fits at ~1024 px; run FaceDetailer and upscaling as separate queue runs if memory runs out.
- ComfyUI runs with `--offline`: the page cannot reach the internet and paid cloud ("partner") nodes are off.
- Use your own or licensed reference images; FaceID is for consistent fictional characters or people who agreed.
