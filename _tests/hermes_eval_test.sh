#!/usr/bin/env bash
## @brief Offline: hermes_eval.sh task selection, exit codes and output freshness, with a fake hermes and ai.
set -u
K=$(cd "$(dirname "$0")/.." && pwd)/agent-hermes
T=$(mktemp -d); fail=0; ok() { echo "ok   $1"; }; no() { echo "FAIL $1"; fail=1; }
H=$T/home; mkdir -p "$H/work/outputs/comfy/old" "$H/work/projects/hermes-cube/output" "$T/bin"
printf '#!/bin/sh\n[ "$1" = doctor ] && exit 0\necho DONE\n' > "$T/bin/hermes"
printf '#!/bin/sh\necho "  internet: OFF"\n' > "$T/bin/ai"
printf '#!/bin/sh\nexit 7\n' > "$T/bin/curl"
chmod +x "$T"/bin/*
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(200,30,30)).save('$H/work/outputs/comfy/old/hermes-mug_00001_.png')" 2>/dev/null
ev() { HOME=$H PATH=$T/bin:$PATH bash "$K/hermes_eval.sh" "$@" > "$T/ev.out" 2>&1; }
ev 1 7; r=$?; s=$(ls -td "$H"/work/hermes-eval/*/ | head -1)summary.md
grep -q '^| 1 |' "$s" && grep -q '^| 7 |' "$s" && [ "$(grep -c '^| [0-9] |' "$s")" = 2 ] && ok "tasks \"1 7\": exactly those two run" || no "task selection"
[ $r = 1 ] && grep -q "RESULT: FAIL" "$s" && ok "a failed task makes the evaluation exit 1" || no "eval exit code ($r)"
ev 7; [ $? = 0 ] && ok "all selected tasks passed -> exit 0" || no "eval exit 0"
ev 10; [ $? = 2 ] && ok "unknown task number refused" || no "bad selection"
sleep 1; ev 4; [ $? = 1 ] && ok "an old image from before the run gets no credit" || no "freshness"
rm -rf "$T"; exit $fail
