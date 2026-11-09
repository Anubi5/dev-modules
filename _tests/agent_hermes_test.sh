#!/usr/bin/env bash
## @brief Offline: agent-hermes adopts a pinned install, refuses foreign source untouched, rejects a wrong installer.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd); fail=0
case_dir() { d=$(mktemp -d); mkdir -p "$d/setup" "$d/bin" "$d/home"; echo "$d"; }
stubs() { # $1 dir, $2 runtime-check exit code
  cat > "$1/setup/hermes_runtime.py" <<P
import sys
PINNED_REVISION='d923a5eb487a17c0a68801bba9ef0f56c81040a8'
if __name__=='__main__':
    open('$1/log','a').write('runtime '+sys.argv[1]+'\n'); sys.exit($2 if sys.argv[1]=='check' else 0)
P
  for f in hermes_local_setup hermes_profile_check; do
    printf "import sys\nopen('%s/log','a').write('%s\\\\n')\n" "$1" "$f" > "$1/setup/$f.py"; done
  printf '#!/bin/sh\necho "curl $*" >> %s/log; echo fake > "$4"\n' "$1" > "$1/bin/curl"
  printf '#!/bin/sh\necho Hermes 0\n' > "$1/bin/hermes"; chmod +x "$1/bin/"*
}
run() { EVA_SETUP="$1/setup" HERMES_HOME="$1/home" PATH="$1/bin:$PATH" bash "$here/agent-hermes/install.sh" >/dev/null 2>&1; }

d=$(case_dir); stubs "$d" 0; run "$d"; rc=$?
[ $rc -eq 0 ] && ! grep -q curl "$d/log" && grep -q hermes_local_setup "$d/log" && echo "PASS adopt pinned install" || { echo "FAIL adopt ($rc)"; fail=1; }

d=$(case_dir); stubs "$d" 1; mkdir -p "$d/home/hermes-agent"; touch "$d/home/hermes-agent/keep"; run "$d"; rc=$?
[ $rc -ne 0 ] && [ -f "$d/home/hermes-agent/keep" ] && ! grep -qE 'curl|runtime install|hermes_local_setup' "$d/log" \
  && echo "PASS foreign source refused untouched" || { echo "FAIL foreign source ($rc)"; fail=1; }

d=$(case_dir); stubs "$d" 1; run "$d"; rc=$?
[ $rc -ne 0 ] && grep -q curl "$d/log" && ! grep -qE 'runtime install|hermes_local_setup' "$d/log" \
  && echo "PASS wrong installer hash rejected" || { echo "FAIL installer hash ($rc)"; fail=1; }
exit $fail
