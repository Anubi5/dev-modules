#!/usr/bin/env bash
## @brief Stops ComfyUI. Your workflows (~/work/comfyui) and results (~/work/outputs/comfy) stay.
PYTHONPATH="${MOD:-$(cd "$(dirname "$0")" && pwd)}" python3 -B -c 'import catalogue; catalogue.migrate_acceptance()' || exit $?
systemctl --user stop comfyui 2>/dev/null; true
