# comfyui (image and video generation)
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) with SDXL + 4x-UltraSharp and Wan 2.2 5B video (GGUF).
Page: http://127.0.0.1:8188 (`dev comfy`). Agents use `ai gen image|video "prompt"` (skill `gpu-generate`):
it unloads the chat model through the model-serving contract, runs one fixed workflow, frees the GPU.

- Custom nodes: `nodes.txt` · model files: `models.txt` - add a line, push, `dev mod update comfyui`.
- Model files already in D:\AI\models\comfy are used read-only; missing ones download into `$MOD/models`.
- Your saved workflows and settings: `~/work/comfyui/` (kept on removal). Results: `~/work/outputs/comfy/<date>/`.
- `test.sh --full` = SDXL image + upscale and a Wan video through `ai gen`.

## Control, faces, repair, upscaling (in the ComfyUI page)
| Want | Nodes (double-click the canvas, type the name) | Model files (models.txt) |
|---|---|---|
| Pose from a photo, skeleton or Blender render | `DWPose Estimator` → `Apply ControlNet` (strength ~0.65, end ~0.75) | `controlnet-union-sdxl-promax` (one ControlNet for pose, depth, canny, lineart) |
| Depth / layout | `Depth Anything` or `MiDaS` preprocessor → same ControlNet | same |
| Same face across images | `IPAdapter FaceID` (weight ~0.8) with a face reference; lighter: `IPAdapter Advanced` + `ip-adapter-plus-face_sdxl` | FaceID Plus v2 + its LoRA, plus-face, CLIP-ViT-H |
| Fix faces / hands | `FaceDetailer` with `UltralyticsDetectorProvider` (`bbox/face_yolov8m.pt`, `bbox/hand_yolov8s.pt`), denoise ~0.3 | face/hand detectors |
| Big sharp output | `Ultimate SD Upscale` (upscale 2x, tile 768/1024, denoise ~0.2) with 4x-UltraSharp | already there |

- Pose and depth preprocessors and InsightFace (FaceID) download their small helper models on first use - internet on.
- 12 GB GPU: SDXL + ControlNet + IPAdapter fits at ~1024 px; run FaceDetailer and upscaling as separate queue runs if memory runs out.
- ComfyUI runs with `--offline`: the page cannot reach the internet and paid cloud ("partner") nodes are off.
  ComfyUI-Manager still installs nodes, but prefer adding them to `nodes.txt`.
- Use your own or licensed reference images; FaceID is for consistent fictional characters or people who agreed.
