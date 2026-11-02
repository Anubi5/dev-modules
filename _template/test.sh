#!/usr/bin/env bash
## @brief Quick real check (exit 0 = works); --full for the slow one. Replace the line below: CI rejects a test
##        that only says "true".
set -euo pipefail
true   # e.g. "$MOD/app/venv/bin/<tool>" --version
