#!/usr/bin/env bash
## @brief Switches the Hermes Blender plugin off and checks it is off (Blender itself goes with the module folder).
## @return 1 when the plugin is still enabled - the runner then keeps the module, so the removal can be retried
command -v hermes >/dev/null || exit 0                       # no Hermes: nothing to switch off
hermes plugins disable blender </dev/null >/dev/null 2>&1
line=$(hermes plugins list 2>/dev/null | grep -w blender | head -1)
if [[ "$line" == *enabled* && "$line" != *disabled* ]]; then
  echo "  CHECK the Hermes Blender plugin is still enabled - switch it off (hermes plugins disable blender), then remove again"
  exit 1
fi
exit 0
