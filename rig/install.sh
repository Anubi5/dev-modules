#!/usr/bin/env bash
set -euo pipefail
: "${MOD:?module folder is required}"
command -v uv >/dev/null
command -v git >/dev/null
command -v blender >/dev/null || { echo "Install the Blender module first" >&2; exit 1; }
mkdir -p "$MOD/app/cache" "$MOD/app/tmp"
export UV_CACHE_DIR="$MOD/app/cache/uv" TMPDIR="$MOD/app/tmp" XDG_CACHE_HOME="$MOD/app/cache"
export HF_HOME="$MOD/app/cache/huggingface" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
REV=6793c6640ff01c8fb389f3993434124bb43d2933
SRC="$MOD/app/unirig-$REV"
if [ ! -d "$SRC/.git" ]; then
  if [ -n "${RIG_SOURCE_DIR:-}" ]; then
    [ "$(git -C "$RIG_SOURCE_DIR" rev-parse HEAD)" = "$REV" ] || exit 1
    git clone -q --no-hardlinks --no-checkout "$RIG_SOURCE_DIR" "$SRC"
    git -C "$SRC" remote set-url origin https://github.com/VAST-AI-Research/UniRig
  else
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/VAST-AI-Research/UniRig
    git -C "$SRC" fetch -q --depth 1 origin "$REV"
  fi
  git -C "$SRC" checkout -q --detach "$REV"
  git -C "$SRC" submodule update --init --recursive
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$REV" ] || exit 1
PY=$(uv python find 3.11 2>/dev/null || true)
if [ -z "$PY" ]; then
  export UV_PYTHON_INSTALL_DIR="$MOD/app/python"
  uv python install 3.11
  PY=$(uv python find 3.11)
fi
VENV="$MOD/app/venv"
[ -x "$VENV/bin/python" ] || uv venv -q --python "$PY" "$VENV"
FLASH='https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
# Unchanged pins are not reinstalled: uv treats the direct-URL wheel as new on every run (its local version tag
# differs in letter case), which reinstalled flash-attn and bpy on each update.
STAMP="$VENV/.eva-pins"; WANT=$(cat "$MOD/requirements.lock" | sha256sum | cut -d' ' -f1)-$(printf '%s' "$FLASH" | sha256sum | cut -c1-16)
if [ "$(cat "$STAMP" 2>/dev/null)" = "$WANT" ] && "$VENV/bin/python" -c 'import torch, bpy, flash_attn' 2>/dev/null; then
  echo "Rig runtime unchanged (pinned packages already installed)."
else
  uv pip install --python "$VENV/bin/python" --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html \
    --only-binary torch,torchvision,bpy,spconv-cu121,torch-scatter,torch-cluster,flash-attn \
    -r "$MOD/requirements.lock" "$FLASH"
  printf '%s' "$WANT" > "$STAMP"
fi
"$VENV/bin/python" "$MOD/fetch_assets.py"
printf '%s\n' "$SRC" > "$MOD/app/source-path.txt"
chmod +x "$MOD/bin/ai-rig" "$MOD/test.sh"
echo "Rig runtime and pinned assets prepared."
