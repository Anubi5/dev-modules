#!/usr/bin/env bash
## @file hermes_test.sh
## @brief Checks that Hermes answers on the local model AND uses a tool: it must write a marker file.
## @details Run as the dev user (AI.ps1 -Action Test), through the normal ai hermes launcher.
## @return 0 if ~/work/hermes-test.txt contains HERMES_TOOL_OK.
export PATH=$HOME/.local/bin:$PATH
export HERMES_DISABLE_LAZY_INSTALLS=1 UV_OFFLINE=1 PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
systemctl is-active -q net-off || { echo 'Hermes acceptance requires internet OFF; nothing changed' >&2; exit 1; }
python3 -B "$(dirname "$(readlink -f "$0")")/hermes_profile_check.py" || exit $?
cd ~/work || exit 1
rm -f hermes-test.txt
q='Use your terminal tool to run exactly this command: echo HERMES_TOOL_OK > ~/work/hermes-test.txt  Then reply DONE.'
timeout 900 ai hermes chat -Q -q "$q" 2>&1 | tail -5
grep -q HERMES_TOOL_OK hermes-test.txt 2>/dev/null
