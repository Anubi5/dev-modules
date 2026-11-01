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
- Model files already in D:\AI\models\comfy are used read-only; missing ones download into `$MOD/models`.
- `models.txt` takes Hugging Face lines and direct-download lines (`<https URL> <file> <folder> sha256:<hex>`, e.g. Juggernaut
  XL Ragnarok from CivitAI); a direct download is kept only if its SHA-256 matches. Models and their tuned settings live in
  `profiles.json` (`ai gen image ... --model realvis|juggernaut|sdxl-base`, `--style photo|art`); `--checkpoint <file>` uses
  any other SDXL file (checked to be SDXL). Every result gets a `.json` record (prompt, seed, model, settings, time, memory);
  with repair/upscale options the untouched original is kept as `<name>_base_*`.
- `ai gen schema` saves ComfyUI's node definitions (`~/work/outputs/comfy/schema.json`); committed as `schema.json`, the
  GitHub check validates every workflow variant against it (`check_workflows.py`) without a GPU.
- Your saved workflows and settings: `~/work/comfyui/` (kept on removal). Results: `~/work/outputs/comfy/<date>/`.
- `test.sh --full` = SDXL image + upscale, a Wan video, and a pose + face + face/hand repair + tiled-upscale job, all through `ai gen`.

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
