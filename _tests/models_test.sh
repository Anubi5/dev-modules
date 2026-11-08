#!/usr/bin/env bash
## @file models_test.sh
## @brief Model-file placement tests with a fake huggingface_hub (no network, no GPU): two copies of a file are only
##        merged when their SHA-256 matches; any difference (same or other size) fails and keeps both; moves to D: keep
##        nothing in the distro. Covers comfyui/fetch_models.py and llama-swap/resolve_models.py.
## @return 0 when every case passes
set -u
R=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); fail=0
ok() { echo "ok   $1"; }; no() { echo "FAIL $1"; fail=1; }
mkdir -p "$T/stub/huggingface_hub"
cat > "$T/stub/huggingface_hub/__init__.py" <<'PY'
class HfApi:
    def list_repo_files(self, repo): return ["m-Q4_K_M.gguf"]
    def model_info(self, repo): return 1
def hf_hub_download(*a, **k): raise RuntimeError("no network in tests")
PY
export PYTHONPATH=$T/stub
# --- comfyui ---
C=$T/c; mkdir -p "$C/ro/checkpoints" "$C/st/checkpoints"
echo SAME > "$C/ro/checkpoints/dup.safetensors"; echo SAME > "$C/st/checkpoints/dup.safetensors"
echo AAAA > "$C/ro/checkpoints/eq.safetensors";  echo BBBB > "$C/st/checkpoints/eq.safetensors"
echo AAAA > "$C/ro/checkpoints/sz.safetensors";  echo BBBBBBBB > "$C/st/checkpoints/sz.safetensors"
for f in dup eq sz; do echo "r $f.safetensors checkpoints"; done > "$C/list.txt"
python3 "$R/comfyui/fetch_models.py" "$C/list.txt" "$C/ro" - "$C/st" "$C/st" > "$C/out" 2>&1; r=$?
[ ! -e "$C/st/checkpoints/dup.safetensors" ] && ok "comfyui: identical copy in the distro removed" || no "comfyui identical"
[ -e "$C/st/checkpoints/eq.safetensors" ] && [ -e "$C/st/checkpoints/sz.safetensors" ] && [ $r -ne 0 ] \
  && [ "$(grep -c 'differs' "$C/out")" = 2 ] && ok "comfyui: differing copies (same and other size) kept, install fails" || { no "comfyui conflicts"; cat "$C/out"; }
mkdir -p "$C/rw"; echo MOVE > "$C/st/checkpoints/mv.safetensors"; echo "r mv.safetensors checkpoints" > "$C/l2.txt"
python3 "$R/comfyui/fetch_models.py" "$C/l2.txt" "$C/rw" "$C/rw" "$C/st" "$C/st" >/dev/null 2>&1
[ -e "$C/rw/checkpoints/mv.safetensors" ] && [ ! -e "$C/st/checkpoints/mv.safetensors" ] && ok "comfyui: in-distro file moved to D:" || no "comfyui move"
# --- llama-swap ---
L=$T/l; mkdir -p "$L/ro/a" "$L/st/a" "$L/ro/b" "$L/st/b"
echo 1111 > "$L/ro/a/m-Q4_K_M.gguf"; echo 22222222 > "$L/st/a/m-Q4_K_M.gguf"     # other size
echo 3333 > "$L/ro/b/m-Q4_K_M.gguf"; echo 3333 > "$L/st/b/m-Q4_K_M.gguf"         # identical
echo '{"chat":[{"id":"a","repo":"r/a","quant":["Q4_K_M"]},{"id":"b","repo":"r/b","quant":["Q4_K_M"]}]}' > "$L/cfg.json"
python3 -c "
import json; S='$L'; json.dump({i:{'repo':'r/'+i,'file':S+'/st/'+i+'/m-Q4_K_M.gguf','files':[S+'/st/'+i+'/m-Q4_K_M.gguf'],'quant':['Q4_K_M']} for i in 'ab'}, open(S+'/st/models.json','w'))"
python3 "$R/llama-swap/resolve_models.py" --config "$L/cfg.json" --state "$L/st" --ro "$L/ro" > "$L/out" 2>&1; r=$?
[ -e "$L/st/a/m-Q4_K_M.gguf" ] && grep -q "a FAILED" "$L/out" && [ $r -ne 0 ] && ok "llama-swap: differing copy kept, model fails" || { no "llama conflict"; cat "$L/out"; }
[ ! -e "$L/st/b/m-Q4_K_M.gguf" ] && grep -q "$L/ro/b" "$L/st/models.json" && ok "llama-swap: identical copy removed, D: path recorded" || no "llama identical"
# The real ComfyUI quick-test parser must accept saved-as + checksum + optional independently.
Q=$T/quick; mkdir -p "$Q/root/comfy/checkpoints"
cat > "$Q/models.txt" <<'LIST'
r upstream.safetensors checkpoints renamed.safetensors sha256:0000000000000000000000000000000000000000000000000000000000000000
r plain.safetensors checkpoints sha256:0000000000000000000000000000000000000000000000000000000000000000
r optional.safetensors checkpoints renamed-optional.safetensors sha256:0000000000000000000000000000000000000000000000000000000000000000 optional
LIST
touch "$Q/root/comfy/checkpoints/renamed.safetensors" "$Q/root/comfy/checkpoints/plain.safetensors"
{
  echo 'miss=0'
  sed -n '/^while read -r repo want folder/,/^done < "\$MOD\/models.txt"/p' "$R/comfyui/test.sh"
  echo 'exit $miss'
} > "$Q/check.sh"
MOD="$Q" MODELS_MOUNT="$Q/root" bash "$Q/check.sh" > "$Q/out" 2>&1 \
  && ok "comfy quick test: saved-as, pin and optional are separate fields" || { no "comfy quick manifest parser"; cat "$Q/out"; }
rm "$Q/root/comfy/checkpoints/renamed.safetensors"
if MOD="$Q" MODELS_MOUNT="$Q/root" bash "$Q/check.sh" > "$Q/out" 2>&1; then no "comfy quick test missed absent required file"
else ok "comfy quick test: missing required renamed file still fails"; fi
bash "$R/_tests/reverse_test.sh" || no "reverse-tools scoped readiness"
python3 -B -m unittest discover -s "$R/llama-swap" -p 'test_variants.py' || no "shared-weight model variants"
rm -rf "$T"; exit $fail
