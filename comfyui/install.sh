#!/usr/bin/env bash
## @brief Installs or updates ComfyUI into $MOD/app/ComfyUI: venv, PyTorch (CUDA), custom nodes, model paths, model files.
set -euo pipefail
cd "$MOD"; C=$MOD/app/ComfyUI; mkdir -p "$MOD/app" "$MOD/models" ~/work/comfyui ~/work/outputs/comfy ~/work/inputs ~/work/scratch/comfy
[ -d "$C" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$C"
cd "$C"
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv -r requirements.txt huggingface_hub
cd custom_nodes
grep -vE '^\s*(#|$)' "$MOD/nodes.txt" | while read -r r; do
  n=$(basename "$r" .git); [ -d "$n" ] || git clone --depth 1 "$r" "$n"
  [ -f "$n/requirements.txt" ] && uv pip install -q --python ../.venv -r "$n/requirements.txt"
done
cd "$C"
printf '%s\n' 'windows:' "  base_path: ${MODELS_MOUNT:-/models}/comfy" '  checkpoints: checkpoints' '  diffusion_models: diffusion_models' \
  '  unet: diffusion_models' '  vae: vae' '  text_encoders: text_encoders' '  clip: text_encoders' '  loras: loras' '  upscale_models: upscale_models' \
  '  controlnet: controlnet' 'module:' "  base_path: $MOD/models" '  checkpoints: checkpoints' '  diffusion_models: diffusion_models' \
  '  unet: diffusion_models' '  vae: vae' '  text_encoders: text_encoders' '  clip: text_encoders' '  loras: loras' '  upscale_models: upscale_models' \
  '  controlnet: controlnet' > extra_model_paths.yaml
for d in ~/work/comfyui/default/ComfyUI-Manager ~/work/comfyui/__manager; do   # Manager: no installs from arbitrary git URLs
  mkdir -p "$d"; printf '[default]\nsecurity_level = normal\nnetwork_mode = public\n' > "$d/config.ini"
done
.venv/bin/python "$MOD/fetch_models.py" "$MOD/models.txt" "${MODELS_MOUNT:-/models}/comfy" "$MOD/models"
echo "ComfyUI ready"
