#!/usr/bin/env bash
## @brief Quick: ComfyUI answers, the custom nodes loaded and the model files exist; --full: SDXL + upscale and a Wan video through ai gen.
set -uo pipefail
systemctl --user start comfyui
for _ in $(seq 90); do curl -sf -m 3 http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
curl -sf -m 3 http://127.0.0.1:8188/system_stats >/dev/null || { echo "ComfyUI not answering (ai logs comfyui)"; exit 1; }
miss=0
nodes=$(curl -sf -m 30 http://127.0.0.1:8188/object_info) || { echo "ComfyUI node list not answering"; exit 1; }
for c in UnetLoaderGGUF DWPreprocessor ControlNetApplyAdvanced IPAdapterAdvanced IPAdapterFaceID FaceDetailer UltralyticsDetectorProvider UltimateSDUpscale; do
  echo "$nodes" | jq -e --arg c "$c" 'has($c)' >/dev/null || { echo "  node $c not loaded (ai logs comfyui: look for 'import failed')"; miss=1; }
done
while read -r repo want folder name; do
  case "$repo" in ''|\#*) continue ;; esac
  name=${name:-$(basename "$want")}
  [ -e "${MODELS_MOUNT:-/models}/comfy/$folder/$name" ] || [ -e "$MOD/models/$folder/$name" ] || { echo "  missing $folder/$name"; miss=1; }
done < "$MOD/models.txt"
[ $miss = 0 ] || exit 1
echo "  ComfyUI up, custom nodes loaded, all model files present"
[ "${1:-}" = --full ] && { cd "$MOD" && exec python3 comfy_test.py; }
exit 0
