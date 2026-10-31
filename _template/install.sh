#!/usr/bin/env bash
## @brief Installs or updates <name> into $MOD (safe to run again). Runs as dev.
set -euo pipefail
cd "$MOD"
# example: [ -x "$MOD/app/venv/bin/python" ] || python3 -m venv "$MOD/app/venv"
# "$MOD/app/venv/bin/pip" install -q -U <package>
