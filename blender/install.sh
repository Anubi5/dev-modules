#!/usr/bin/env bash
## @brief Downloads the newest official Blender into $MOD/app (skipped when present) and provides the blender command.
set -euo pipefail
R=https://download.blender.org/release/; mkdir -p "$MOD/app" "$MOD/bin"
if ! ls -d "$MOD"/app/blender-*-linux-x64 >/dev/null 2>&1; then
  ver=$(curl -s "$R" | grep -oE 'Blender[0-9]+\.[0-9]+/' | sort -V | uniq | tail -1)
  tb=$(curl -s "$R$ver" | grep -oE 'blender-[0-9.]+-linux-x64\.tar\.xz' | sort -V | uniq | tail -1)
  curl -sL "$R$ver$tb" | tar -xJ -C "$MOD/app"
fi
latest=$(ls -d "$MOD"/app/blender-*-linux-x64 | sort -V | tail -1)
ln -sfn "$latest" "$MOD/app/current"
"$MOD/app/current/blender" -b --version | head -1
