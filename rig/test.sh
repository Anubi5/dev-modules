#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHONDONTWRITEBYTECODE=1 python3 "$HERE/test_contract.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$HERE/cli.py" --help >/dev/null
if [ "${1:-}" = --full ]; then
  : "${RIG_TEST_INPUT:?set RIG_TEST_INPUT to a fixed adult fictional humanoid GLB fixture}"
  bash "$HERE/bin/ai-rig" "$RIG_TEST_INPUT" --name rig-acceptance --seed 12345
fi
