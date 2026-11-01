#!/usr/bin/env bash
## @brief Quick: Blender starts; --full: renders a small scene on the GPU through the blender command (GPU lock + ai free).
set -uo pipefail
"$MOD/app/current/blender" -b --version >/dev/null 2>&1 || { echo "blender does not start"; exit 1; }
[ "${1:-}" = --full ] || { echo "  Blender starts"; exit 0; }
cd "$MOD" && "$MOD/bin/blender" -b --factory-startup --python-exit-code 1 --python blender_test.py 2>&1 | grep 'BLENDER-TEST OK'
