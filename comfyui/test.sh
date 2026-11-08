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
while read -r repo want folder name tags; do
  case "$repo" in ''|\#*) continue ;; esac
  case " ${name:-} ${tags:-} " in *' optional '*) continue ;; esac # optional models are reported by ai gen models
  case "${name:-}" in ''|sha256:*) name=$(basename "$want") ;; esac
  [ -e "${MODELS_MOUNT:-/models}/comfy/$folder/$name" ] || [ -e "$MOD/models/$folder/$name" ] || { echo "  missing $folder/$name"; miss=1; }
done < "$MOD/models.txt"
echo "  image models:"; python3 "$MOD/gen.py" models || miss=1     # every model in profiles.json: ready / optional / MISSING
# preflight against the live ComfyUI: every model file of every workflow variant is in its loader's list (test LoRAs excluded)
pf=$(cd "$MOD" && python3 -c '
import gen
bad = sorted({m for wf in gen.all_workflows() for m in gen.missing_files(wf) if not m.startswith("LoraLoader.")})
print("\n".join(bad))') || { echo "  CHECK preflight could not run"; miss=1; }
[ -z "$pf" ] && echo "  preflight: ComfyUI sees every model file the workflows name" \
  || echo "$pf" | sed 's/^/  (preflight would refuse, fine if that model is optional and not installed: /; s/$/)/'
n=$(python3 "$MOD/gen.py" list --json | jq '[.models[], .checkpoints[], .loras[]] | length') \
  && echo "  ai gen list: $n models, checkpoints and LoRAs found" || { echo "  CHECK ai gen list failed"; miss=1; }
ca=$(cd "$MOD" && python3 -c '
import gen
m = gen.class_arch()
left = sorted(gen.archs() - set(m.values()))
print(("OK " if not left else "MISSING ") + ", ".join(f"{c}={a}" for c, a in sorted(m.items())) + (" | not recognised: " + ", ".join(left) if left else ""))')
case "$ca" in OK*) echo "  model detection (ComfyUI's own): ${ca#OK }" ;; *) echo "  CHECK model detection: $ca"; miss=1 ;; esac
[ $miss = 0 ] || exit 1
echo "  ComfyUI up, custom nodes loaded, all model files present"
# img-check (agents quote it to prove a result; the 3D check uses it on the renders): each case must get its verdict
V=$MOD/app/ComfyUI/.venv/bin/python; t=$(mktemp -d)
"$V" - "$t" <<'PY'
import sys, numpy as np; from PIL import Image, ImageDraw
d = sys.argv[1]; S = lambda n, i: i.save(f"{d}/{n}.png")
S("solid_red", Image.new("RGB", (64, 64), (200, 30, 30))); S("solid_blue", Image.new("RGB", (64, 64), (30, 30, 200)))
S("transparent", Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
S("near_black", Image.fromarray(np.random.default_rng(1).integers(0, 4, (64, 64, 3), dtype=np.uint8)))
g = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1)); S("gradient", Image.fromarray(np.stack([g, g // 2, 255 - g], -1)))
i = Image.new("RGB", (64, 64), (60, 60, 60)); ImageDraw.Draw(i).rectangle((16, 16, 48, 48), fill=(200, 30, 30)); S("square", i)
for n, box in (("object", (16, 48)), ("small_object", (60, 72))):      # a shaded shape on transparency: 26% / 1% cover
    i = Image.new("RGBA", (64, 64) if n == "object" else (120, 120), (0, 0, 0, 0)); dr = ImageDraw.Draw(i)
    for y in range(*box):
        dr.line((box[0], y, box[1], y), fill=(100 + 150 * (y - box[0]) // (box[1] - box[0]), 80, 40, 255))
    S(n, i)
S("opaque_flat", Image.new("RGBA", (64, 64), (200, 30, 30, 255)))     # all visible, one colour: the per-channel rule
i = Image.new("RGBA", (200, 200), (0, 0, 0, 0)); i.putpixel((5, 5), (255, 0, 0, 255)); i.putpixel((6, 5), (0, 0, 255, 255))
S("speck", i)                                                          # varied but almost nothing visible: the coverage rule
PY
ic() { "$MOD/bin/img-check" "$@" >/dev/null; echo $?; }
res=""; for c in solid_red:1 solid_blue:1 opaque_flat:1 speck:1 transparent:1 near_black:1 gradient:0 square:0 object:0; do
  [ "$(ic "$t/${c%:*}.png")" = "${c#*:}" ] || res="$res ${c%:*}"; done
[ "$(ic --min-coverage 0.02 "$t/small_object.png")" = 1 ] && [ "$(ic --min-coverage 0.02 "$t/object.png")" = 0 ] || res="$res coverage"
[ -z "$res" ] && echo "  img-check: solid colours, specks, transparent and near-black -> BLANK; gradient, square, object pass; small object too small" \
  || { echo "img-check gives the wrong verdict for:$res"; "$MOD/bin/img-check" "$t"/*.png; rm -rf "$t"; exit 1; }
rm -rf "$t"
[ "${1:-}" = --full ] && { cd "$MOD" && exec python3 comfy_test.py; }
exit 0
