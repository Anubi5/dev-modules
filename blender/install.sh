#!/usr/bin/env bash
## @brief Blender (pinned, checksum-checked) + the official Blender Lab MCP add-on + the official Hermes Blender plugin.
## @details Agents work in Blender through the official MCP server (Blender Lab, pinned commit), not through our own
##          scripts. A private virtual display (Xvfb :99, service blender-display) gives Blender a real window inside
##          the sandbox, so screenshots and viewport renders work; nothing appears on Windows.
set -euo pipefail
VER=${BLENDER_VERSION:-5.2.2}                                       # pinned; change on purpose, the old one stays
ADDON_VER=1.0.3; ADDON_SHA=a7a9da816192502e5a0a202a396444e266b47d8fc4f74ad4698048bd43040707
MCP_SRC="git+https://projects.blender.org/lab/blender_mcp.git@2cea8d566dde07fbac28a61d698909d69724e853#subdirectory=mcp"
MCP_PYTHON=3.14.7
say() { echo "  [$(date +%H:%M:%S)] $*"; }
R=https://download.blender.org/release/Blender${VER%.*}/; TB=blender-$VER-linux-x64.tar.xz
mkdir -p "$MOD/app" "$MOD/bin"
if [ ! -x "$MOD/app/blender-$VER-linux-x64/blender" ]; then
  say "downloading Blender $VER (~400 MB, checked against blender.org's sha256 list)"
  curl -sfL -o "$MOD/app/$TB" "$R$TB"
  want=$(curl -sfL "${R}blender-$VER.sha256" | awk -v f="$TB" '$2 == f {print $1}')
  [ -n "$want" ] && [ "$(sha256sum "$MOD/app/$TB" | cut -d' ' -f1)" = "$want" ] \
    || { rm -f "$MOD/app/$TB"; echo "  CHECK Blender $VER download does not match blender.org's checksum - not installed"; exit 1; }
  tar -xJf "$MOD/app/$TB" -C "$MOD/app"; rm -f "$MOD/app/$TB"
fi
ln -sfn "$MOD/app/blender-$VER-linux-x64" "$MOD/app/current"
B=$MOD/app/current/blender; v=$("$B" -b --version 2>/dev/null || true); echo "${v%%$'\n'*}"

addon_ok() { local o; o=$("$B" -b --online-mode --command blender_mcp --help 2>&1 || true); [[ "$o" == *"usage: blender_mcp"* ]]; }
say "Blender Lab MCP add-on $ADDON_VER"
if ! addon_ok; then
  z=$MOD/app/mcp-$ADDON_VER.zip
  curl -sfL -o "$z" "https://projects.blender.org/lab/blender_mcp/releases/download/v$ADDON_VER/mcp-$ADDON_VER.zip"
  [ "$(sha256sum "$z" | cut -d' ' -f1)" = "$ADDON_SHA" ] || { rm -f "$z"; echo "  CHECK add-on checksum mismatch - not installed"; exit 1; }
  "$B" -b --command extension install-file -r user_default -e "$z" >/dev/null && rm -f "$z"
fi
addon_ok || { echo "  CHECK MCP add-on not working"; exit 1; }

say "MCP server (Blender Lab, pinned commit) - prepared now so it starts offline later"
# A repeat install (also offline) reuses the prepared runtime when its source pin, lock, launcher, provenance and
# dependency metadata are unchanged (not a hash of every installed package file); otherwise uv resolves it again.
# "prepare" below re-validates the same records either way.
if MCP_BIN=$(python3 -B "$MOD/mcp_runtime.py" reuse --module "$MOD" --source "$MCP_SRC" 2>/dev/null); then
  echo "  prepared runtime unchanged (same pin and lock): reused"
else
  MCP_BIN=$(uv tool run --python "$MCP_PYTHON" --with-requirements "$MOD/mcp-requirements.lock" \
    --from "$MCP_SRC" python -c 'import shutil; print(shutil.which("blender-mcp") or "")')
fi
[ -x "$MCP_BIN" ] || { echo "  CHECK MCP server could not be prepared"; exit 1; }

say "Hermes Blender plugin (official, catalog pin)"
export PATH="$MOD/app/current:$PATH"   # the plugin installer looks for Blender on PATH; the command link comes after this script
if command -v hermes >/dev/null; then
  pl=$(hermes plugins list 2>/dev/null || true)          # (no pipe into grep -q: with pipefail that can misfire)
  if [[ "$pl" != *" blender "* ]]; then
    hermes plugins install blender </dev/null || { echo "  CHECK hermes plugins install blender failed - Blender not ready"; exit 1; }
  fi
  hermes plugins enable blender </dev/null >/dev/null 2>&1 || { echo "  CHECK hermes plugins enable blender failed - Blender not ready"; exit 1; }
  pl=$(hermes plugins list 2>/dev/null || true)
  row=$(printf '%s\n' "$pl" | grep -w blender | head -1 || true)
  [[ "$row" =~ (^|[^a-z])enabled ]] || { echo "  CHECK the Blender plugin is not listed as enabled in Hermes - Blender not ready"; exit 1; }
  echo "$row"
else
  echo "  Hermes not installed: Blender plugin skipped (dev mod add agent-hermes, then dev mod update blender)"
fi
python3 -B "$MOD/mcp_runtime.py" prepare --module "$MOD" --runtime "$MCP_BIN" --source "$MCP_SRC" \
  --plugin "${HERMES_HOME:-$HOME/.hermes}/plugins/blender"
echo "Blender ready"
