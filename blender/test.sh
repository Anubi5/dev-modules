#!/usr/bin/env bash
## @brief Quick: Blender starts, the MCP add-on works, and the official MCP path works end to end (GUI Blender on the
##        virtual display: scene read + screenshot). --full: also a GPU render through the blender command.
set -uo pipefail
B=$MOD/app/current/blender
MCP_SRC="git+https://projects.blender.org/lab/blender_mcp.git@2cea8d566dde07fbac28a61d698909d69724e853#subdirectory=mcp"
"$B" -b --version >/dev/null 2>&1 || { echo "blender does not start"; exit 1; }
o=$("$B" -b --online-mode --command blender_mcp --help 2>&1 || true); [[ "$o" == *"usage: blender_mcp"* ]] || { echo "MCP add-on missing"; exit 1; }
echo "  Blender starts, MCP add-on installed"
port() { timeout 2 bash -c '</dev/tcp/127.0.0.1/9876' 2>/dev/null; }
started=""
if ! port; then                                   # nobody's bridge running: open Blender on the virtual display
  systemctl --user start blender-display || { echo "virtual display (blender-display) does not start"; exit 1; }
  DISPLAY=:99 "$B" --online-mode >/tmp/blender-mcp-test.log 2>&1 & started=$!
  for _ in $(seq 60); do port && break; sleep 1; done
fi
port || { echo "MCP bridge did not open port 9876 (see /tmp/blender-mcp-test.log)"; [ -n "$started" ] && kill $started; exit 1; }
python3 "$MOD/mcp_smoke.py" "$MCP_SRC"; rc=$?
[ -n "$started" ] && { kill $started 2>/dev/null; wait $started 2>/dev/null; }
[ $rc = 0 ] || { echo "official MCP path failed (see above)"; exit 1; }
[ "${1:-}" = --full ] || exit 0
cd "$MOD" && "$MOD/bin/blender" -b --factory-startup --python-exit-code 1 --python blender_test.py 2>&1 | grep 'BLENDER-TEST OK'
