#!/usr/bin/env bash
## @brief Quick: ComfyUI answers, the custom nodes loaded and the model files exist; --full: SDXL + upscale, a Wan video, and pose + face + repair + tiled upscale through ai gen.
set -uo pipefail
systemctl --user start comfyui
for _ in $(seq 90); do curl -sf -m 3 http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
curl -sf -m 3 http://127.0.0.1:8188/system_stats >/dev/null || { echo "ComfyUI not answering (ai logs comfyui)"; exit 1; }
miss=0
nodes=$(curl -sf -m 30 http://127.0.0.1:8188/object_info) || { echo "ComfyUI node list not answering"; exit 1; }
while read -r feature c; do                      # feature -> one node it needs
  echo "$nodes" | jq -e --arg c "$c" 'has($c)' >/dev/null || { echo "  CHECK $feature unavailable: node $c not loaded (ai logs comfyui, look for 'import failed')"; miss=1; }
done <<'LIST'
video(GGUF) UnetLoaderGGUF
pose/depth-control DWPreprocessor
consistent-faces IPAdapterAdvanced
face-repair FaceDetailer
face/hand-detectors UltralyticsDetectorProvider
tiled-upscale UltimateSDUpscale
flux3-edit ReferenceMultiLatent
anima-pose AnimaLLLiteApply
zimage ModelSamplingAuraFlow
fibo2-edit TextEncodeFibo2
LIST
PY=$MOD/app/ComfyUI/.venv/bin/python
"$PY" -c 'import insightface' 2>/dev/null || echo "  (FaceID unavailable: InsightFace not installed - plain IPAdapter face works)"
while read -r repo want folder name; do
  case "$repo" in ''|\#*) continue ;; esac
  case "${name:-}" in *optional*) continue ;; esac              # optional models: reported by "ai gen models" below
  case "${name:-}" in ''|sha256:*) name=$(basename "$want") ;; esac
  [ -e "${MODELS_MOUNT:-/models}/comfy/$folder/$name" ] || [ -e "$MOD/models/$folder/$name" ] || { echo "  missing $folder/$name"; miss=1; }
done < "$MOD/models.txt"
echo "  image models:"; python3 "$MOD/gen.py" models || miss=1     # every model in profiles.json: ready / optional / MISSING
[ $miss = 0 ] || exit 1
echo "  ComfyUI up, custom nodes loaded, all model files present"
# img-check (agents quote it to prove a result): a flat image must be called BLANK, a real one must pass
V=$MOD/app/ComfyUI/.venv/bin/python; t=$(mktemp -d)
"$V" -c "from PIL import Image, ImageDraw; Image.new('RGB',(64,64),(60,60,60)).save('$t/flat.png'); i=Image.new('RGB',(64,64),(60,60,60)); ImageDraw.Draw(i).rectangle((16,16,48,48),fill=(200,30,30)); i.save('$t/cube.png')"
flat=$("$MOD/bin/img-check" "$t/flat.png"); fr=$?; cube=$("$MOD/bin/img-check" "$t/cube.png"); cr=$?   # (exit 1 = blank, by design)
[ $fr = 1 ] && [[ "$flat" == *BLANK* ]] && [ $cr = 0 ] && [[ "$cube" == *red* ]] \
  && echo "  img-check: flat image -> BLANK, red square -> red" || { echo "img-check does not work: $flat (exit $fr) / $cube (exit $cr)"; rm -rf "$t"; exit 1; }
rm -rf "$t"
[ "${1:-}" = --full ] && { cd "$MOD" && exec python3 comfy_test.py; }
exit 0
