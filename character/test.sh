#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$HERE" -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 "$HERE/cli.py" --help >/dev/null
