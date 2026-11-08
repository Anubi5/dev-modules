#!/usr/bin/env bash
# Exercise the real public dispatcher, including an older chat-only command on PATH.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'SH'
#!/usr/bin/env bash
echo '{"data":[{"id":"legacy-chat"}]}'
SH
chmod +x "$T/bin/curl"
if [ -n "${NATIVE_PYTHON:-}" ]; then
  printf '#!/usr/bin/env bash\nexec "$NATIVE_PYTHON" "$@"\n' > "$T/bin/python3"
  chmod +x "$T/bin/python3"
fi
export PATH="$T/bin:/usr/bin:/bin"
# Help and invalid verbs must not query the model server, even without a catalogue.
for flag in help -h --help; do bash "$R/llama-swap/bin/ai-models" "$flag" | grep -q 'ai models'; done
if bash "$R/llama-swap/bin/ai-models" typo 2>"$T/error"; then exit 1; fi
grep -q 'Unknown ai models command' "$T/error"
result=$(bash "$R/llama-swap/bin/ai-models")
result=${result%$'\r'} # Native Windows Python uses CRLF; production Linux output is unchanged.
[ "$result" = '  legacy-chat' ]
if bash "$R/llama-swap/bin/ai-models" show fixture --json 2>"$T/error"; then
  echo 'FAIL absent catalogue silently accepted'; exit 1
fi
grep -q 'needs the comfyui module' "$T/error"
cat > "$T/bin/ai-model-catalogue" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@"
SH
chmod +x "$T/bin/ai-model-catalogue"
for operation in list show verify; do
  result=$(bash "$R/llama-swap/bin/ai-models" "$operation" fixture --json)
  [ "$result" = "$(printf '%s\n' "$operation" fixture --json)" ]
done
[ ! -e "$R/comfyui/bin/ai-models" ]
echo 'OK legacy chat output preserved; inventory arguments forwarded; no duplicate command owner'
