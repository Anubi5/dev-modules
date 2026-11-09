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
MCP_BIN=$(uv tool run --python "$MCP_PYTHON" --with-requirements "$MOD/mcp-requirements.lock" \
  --from "$MCP_SRC" python -c 'import shutil; print(shutil.which("blender-mcp") or "")')
[ -x "$MCP_BIN" ] || { echo "  CHECK MCP server could not be prepared"; exit 1; }

say "Hermes Blender plugin (official, catalog pin)"
if command -v hermes >/dev/null; then
  pl=$(hermes plugins list 2>/dev/null || true)          # (no pipe into grep -q: with pipefail that can misfire)
  [[ "$pl" == *" blender "* ]] || hermes plugins install blender </dev/null || echo "  CHECK hermes plugins install blender failed"
  hermes plugins enable blender </dev/null >/dev/null 2>&1 || true
  hermes plugins list 2>/dev/null | grep -w blender | head -1
fi
python3 -B "$MOD/mcp_runtime.py" prepare --module "$MOD" --runtime "$MCP_BIN" --source "$MCP_SRC" \
  --plugin "${HERMES_HOME:-$HOME/.hermes}/plugins/blender"
echo "Blender ready"
