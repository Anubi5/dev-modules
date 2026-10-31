# comfyui (image and video generation)
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) with SDXL + 4x-UltraSharp and Wan 2.2 5B video (GGUF).
Page: http://127.0.0.1:8188 (`dev comfy`). Agents use `ai gen image|video "prompt"` (skill `gpu-generate`):
it unloads the chat model through the model-serving contract, runs one fixed workflow, frees the GPU.

- Custom nodes: `nodes.txt` · model files: `models.txt` - add a line, push, `dev mod update comfyui`.
- Model files already in D:\AI\models\comfy are used read-only; missing ones download into `$MOD/models`.
- Your saved workflows and settings: `~/work/comfyui/` (kept on removal). Results: `~/work/outputs/comfy/<date>/`.
- `test.sh --full` = SDXL image + upscale and a Wan video through `ai gen`.
