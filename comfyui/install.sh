#!/usr/bin/env bash
## @brief Installs or updates ComfyUI into $MOD/app/ComfyUI: pinned app + custom nodes (nodes.txt), venv with locked
##        Python packages, model paths, model files and the helper models the nodes would otherwise fetch on first use.
## @details Python packages: after the first install their exact versions are kept in app/constraints.txt and every
##          later install sticks to them (a hijacked new release cannot slip in). To upgrade on purpose:
##          COMFY_UPGRADE=1 dev mod update comfyui   (installs current versions and writes a fresh lock).
set -euo pipefail
COMFY_REPO=https://github.com/comfyanonymous/ComfyUI
COMFY_REF=6d352802d8f8b9133f664a1757d6c6d26390c335        # pinned ComfyUI commit (raise it deliberately)
cd "$MOD"; A=$MOD/app; C=$A/ComfyUI; mkdir -p "$A" "$MOD/models" ~/work/comfyui ~/work/outputs/comfy ~/work/inputs ~/work/scratch/comfy

## @brief Puts a git checkout at an exact commit (clones or fetches only that commit). @param $1 url @param $2 commit @param $3 folder
at_commit() {
  [ -d "$3/.git" ] || { mkdir -p "$3"; git -C "$3" init -q; git -C "$3" remote add origin "$1"; }
  [ "$(git -C "$3" rev-parse HEAD 2>/dev/null)" = "$2" ] && return 0
  git -C "$3" fetch -q --depth 1 origin "$2" && git -C "$3" checkout -q --force "$2" \
    && git -C "$3" submodule -q update --init --depth 1 --recursive
}
say() { printf '  [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
say "ComfyUI at the pinned commit ${COMFY_REF:0:10}"
at_commit "$COMFY_REPO" "$COMFY_REF" "$C"
cd "$C"
[ -d .venv ] || uv venv -q --python 3.12 .venv
LOCK=$A/constraints.txt; [ "${COMFY_UPGRADE:-0}" = 1 ] && rm -f "$LOCK"
pin=(); [ -f "$LOCK" ] && pin=(--constraint "$LOCK")
say "Python packages: PyTorch (CUDA) - large the first time"
uv pip install -q --python .venv "${pin[@]}" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
say "Python packages: ComfyUI requirements"
uv pip install -q --python .venv "${pin[@]}" -r requirements.txt huggingface_hub

# custom nodes: "<git url> <commit>" per line; nodes no longer listed are moved to app/removed-nodes/
mkdir -p custom_nodes "$A/removed-nodes"; keep=()
while read -r r ref _; do
  case "$r" in ''|'#'*) continue ;; esac
  [ -n "${ref:-}" ] || { echo "  CHECK $r has no pinned commit in nodes.txt"; exit 1; }
  n=$(basename "$r" .git); keep+=("$n"); say "custom node $n (${ref:0:10})"
  at_commit "$r" "$ref" "custom_nodes/$n" || { echo "  CHECK could not get $n at $ref"; exit 1; }
  if [ -f "custom_nodes/$n/requirements.txt" ]; then uv pip install -q --python .venv "${pin[@]}" -r "custom_nodes/$n/requirements.txt"; fi
done < "$MOD/nodes.txt"
for d in custom_nodes/*/; do n=$(basename "$d"); [ -d "$d/.git" ] || continue
  [[ " ${keep[*]} " == *" $n "* ]] || { mv "$d" "$A/removed-nodes/$n-$(date +%s)"; echo "  node $n no longer listed - moved to app/removed-nodes/"; }; done
say "InsightFace (for FaceID; built from source the first time, a few minutes)"
# FaceID needs InsightFace (built here). Optional: everything else works without it.
.venv/bin/python -c 'import insightface' 2>/dev/null || uv pip install -q --python .venv "${pin[@]}" insightface onnxruntime-gpu \
  || echo "  CHECK InsightFace did not install - IPAdapter FaceID will not work (plain IPAdapter face does)"
[ -f "$LOCK" ] || { uv pip freeze --python .venv > "$LOCK"; echo "  Python package versions locked in $LOCK"; }

roots() {   # model folders, same layout on D: and in the module's own folder (older installs; emptied by the move to D:)
  printf '%s\n' '  checkpoints: checkpoints' '  diffusion_models: diffusion_models' '  unet: diffusion_models' '  vae: vae' \
    '  text_encoders: text_encoders' '  clip: text_encoders' '  loras: loras' '  upscale_models: upscale_models' \
    '  controlnet: controlnet' '  clip_vision: clip_vision' '  ipadapter: ipadapter' '  ultralytics: ultralytics' \
    '  ultralytics_bbox: ultralytics/bbox' '  ultralytics_segm: ultralytics/segm' '  model_patches: model_patches'; }
{ echo 'windows:'; echo "  base_path: ${MODELS_MOUNT:-/models}/comfy"; roots; echo 'module:'; echo "  base_path: $MOD/models"; roots; } > extra_model_paths.yaml

say "model files (models.txt) - each download shows its progress"
# model files live on D: (read-only at /models); D: is writable at /models-rw only while Windows runs "dev mod add|update".
# Older copies inside the distro ($MOD/models) are moved to D: on the way; the checksum record stays in $MOD/models.
.venv/bin/python "$MOD/fetch_models.py" "$MOD/models.txt" "${MODELS_MOUNT:-/models}/comfy" "${MODELS_RW:-/models-rw}/comfy" \
  "$MOD/models" "$MOD/models"
say "helper models (pose detection, InsightFace faces)"
# helper models the nodes would fetch on first use - fetched now, so everything works with the internet off
# files live inside ComfyUI ($C is both where they are looked for and where a missing one is downloaded to)
.venv/bin/python "$MOD/fetch_models.py" "$MOD/helpers.txt" "$C" "$C" "$C"; rm -rf "$C/_dl"
I=$C/models/insightface/models/buffalo_l
if [ ! -f "$I/w600k_r50.onnx" ]; then
  mkdir -p "$I"; curl -sfL -o "$A/buffalo_l.zip" https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && unzip -qo "$A/buffalo_l.zip" -d "$I" && rm -f "$A/buffalo_l.zip" \
    && { [ ! -d "$I/buffalo_l" ] || { mv "$I"/buffalo_l/* "$I"/ && rmdir "$I/buffalo_l"; }; } || echo "  CHECK InsightFace face models (buffalo_l) not fetched"
fi
echo "ComfyUI ready"
