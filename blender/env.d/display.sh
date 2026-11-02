# blender module: GUI programs (Blender) use the private virtual display :99 - screenshots work, nothing shows on Windows
export DISPLAY=:99
systemctl --user is-active -q blender-display 2>/dev/null || systemctl --user start blender-display >/dev/null 2>&1 || true
