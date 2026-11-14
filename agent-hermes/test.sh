#!/usr/bin/env bash
## @brief Offline check: pinned source and launcher, reviewed local profile. No inference, no network.
set -euo pipefail
here=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
export HERMES_DISABLE_LAZY_INSTALLS=1 UV_OFFLINE=1 PIP_NO_INDEX=1 PATH="$HOME/.local/bin:$PATH"
python3 -B "$here/hermes_runtime.py" check >/dev/null && echo "  Hermes source and launcher: pinned"
python3 -B "$here/hermes_profile_check.py" && echo "  Hermes profile: local, reviewed"
echo "OK agent-hermes"
