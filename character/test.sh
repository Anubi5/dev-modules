#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$HERE:$HERE/tests/unit:$HERE/tests/contract:$HERE/tests/integration${PYTHONPATH:+:$PYTHONPATH}"
for category in unit contract integration; do
  [ -d "$HERE/tests/$category" ] || continue
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$HERE/tests/$category" -p 'test_*.py'
done
PYTHONDONTWRITEBYTECODE=1 python3 "$HERE/cli.py" --help >/dev/null
