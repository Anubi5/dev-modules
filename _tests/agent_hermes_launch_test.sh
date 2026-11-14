#!/usr/bin/env bash
## @brief Offline: launch.sh checks the profile before launching, keeps a running chat model, disables lazy installs,
##        passes arguments, stdout and exit code through; hermes_test.sh uses the guarded "ai hermes" route.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)/agent-hermes
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
export HOME="$T/home" PATH="$T/bin:$PATH" CALLS="$T/calls"
mkdir -p "$T/bin" "$HOME/work"
cat > "$T/bin/systemctl" <<'SH'
#!/usr/bin/env bash
if [ "$*" = '--user is-active -q llama-swap' ]; then exit "${LLAMA_RESULT:-0}"; fi
echo "$*" >> "$CALLS"
SH
cat > "$T/bin/python3" <<'SH'
#!/usr/bin/env bash
echo 'profile checked' >> "$CALLS"; echo 'PROFILE_DIAGNOSTIC'; exit "${PROFILE_RESULT:-1}"
SH
cat > "$T/bin/hermes" <<'SH'
#!/usr/bin/env bash
echo 'agent launched' >> "$CALLS"
[ "$HERMES_DISABLE_LAZY_INSTALLS" = 1 ] && [ "$UV_OFFLINE" = 1 ] || exit 99
[ "$(umask)" = 0077 ] || exit 98
[ "$PWD" = "$HOME/work" ] || exit 97
printf '%s\n' "$*" >> "$CALLS"; printf '%s\n' '{"type":"result"}'; exit 7
SH
chmod +x "$T/bin/"*
if bash "$R/launch.sh" >/dev/null 2>&1; then exit 1; fi
grep -q 'profile checked' "$CALLS"; ! grep -q 'agent launched' "$CALLS"
set +e; PROFILE_RESULT=0 bash "$R/launch.sh" chat -Q >"$T/out" 2>"$T/err"; code=$?; set -e
[ "$code" = 7 ]; ! grep -q 'start llama-swap' "$CALLS"; grep -q 'chat -Q' "$CALLS"
grep -q PROFILE_DIAGNOSTIC "$T/err"; [ "$(cat "$T/out")" = '{"type":"result"}' ]
: > "$CALLS"; set +e; LLAMA_RESULT=3 PROFILE_RESULT=0 bash "$R/launch.sh" >/dev/null 2>&1; set -e
grep -q -- '--user start llama-swap' "$CALLS"
echo 'PASS launch: profile before launch; running chat kept, stopped chat started; offline env; args and exit 7 preserved'
cat > "$T/bin/ai" <<'SH'
#!/usr/bin/env bash
[ "$1" = hermes ] && [ "$2" = chat ] && [ "$3" = -Q ] && [ "$4" = -q ] || exit 98
echo 'normal acceptance launcher' >> "$CALLS"; printf '%s\n' HERMES_TOOL_OK > "$HOME/work/hermes-test.txt"
SH
cat > "$T/bin/systemctl" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$T/bin/ai" "$T/bin/systemctl"; : > "$CALLS"
PROFILE_RESULT=0 bash "$R/hermes_test.sh" >/dev/null
grep -q 'normal acceptance launcher' "$CALLS"; ! grep -q 'agent launched' "$CALLS"
echo 'PASS hermes_test.sh uses ai hermes rather than launching the agent directly'
