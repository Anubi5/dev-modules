#!/usr/bin/env bash
set -euo pipefail
: "${MOD:?module folder is required}"
command -v python3 >/dev/null
chmod +x "$MOD/bin/ai-character" "$MOD/test.sh"
echo 'Character file contracts and pipeline tools prepared (no model/runtime installation).'
