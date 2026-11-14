#!/usr/bin/env bash
## @brief Starts Hermes. Called by the platform's "ai hermes" only after its agent gate (internet OFF, probe fails).
## @details Offline environment, private files, reviewed profile, chat model running; arguments and exit code pass through.
set -uo pipefail
here=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
umask 077
export HERMES_DISABLE_LAZY_INSTALLS=1 UV_OFFLINE=1 PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B "$here/hermes_profile_check.py" >&2 || exit $?
systemctl --user is-active -q llama-swap || systemctl --user start llama-swap || exit $?
cd "$HOME/work" || exit 1
exec hermes "$@"
