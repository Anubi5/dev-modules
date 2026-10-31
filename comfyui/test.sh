#!/usr/bin/env bash
## @brief Quick: ComfyUI answers and the model files exist; --full: SDXL + upscale and a Wan video through ai gen.
set -uo pipefail
systemctl --user start comfyui
for _ in $(seq 90); do curl -sf -m 3 http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
curl -sf -m 3 http://127.0.0.1:8188/system_stats >/dev/null || { echo "ComfyUI not answering (ai logs comfyui)"; exit 1; }
miss=0
while read -r repo name folder; do
  case "$repo" in ''|\#*) continue ;; esac
  [ -e "${MODELS_MOUNT:-/models}/comfy/$folder/$name" ] || [ -e "$MOD/models/$folder/$name" ] || { echo "  missing $folder/$name"; miss=1; }
done < "$MOD/models.txt"
[ $miss = 0 ] || exit 1
echo "  ComfyUI up, all model files present"
[ "${1:-}" = --full ] && { cd "$MOD" && exec python3 comfy_test.py; }
exit 0
