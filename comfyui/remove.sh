#!/usr/bin/env bash
## @brief Stops ComfyUI. Your workflows (~/work/comfyui) and results (~/work/outputs/comfy) stay.
systemctl --user stop comfyui 2>/dev/null; true
