#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$HERE" -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 "$HERE/cli.py" --help >/dev/null
if [ "${1:-}" = --full ]; then
  if [ -z "${RIG_TEST_INPUT:-}" ]; then
    echo 'SKIP rig GPU acceptance: set RIG_TEST_INPUT to a fixed adult fictional humanoid GLB fixture; offline checks passed.'
    exit 0
  fi
  [ -s "$RIG_TEST_INPUT" ] || { echo "CHECK RIG_TEST_INPUT is missing or empty: $RIG_TEST_INPUT" >&2; exit 1; }
  bash "$HERE/bin/ai-rig" "$RIG_TEST_INPUT" --name rig-acceptance --seed 12345
fi
