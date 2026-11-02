#!/usr/bin/env bash
## @brief Switches the Hermes Blender plugin off (Blender itself goes with the module folder).
command -v hermes >/dev/null && hermes plugins disable blender </dev/null >/dev/null 2>&1
true
