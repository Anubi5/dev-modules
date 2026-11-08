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
export PATH="$T/bin:/usr/bin:/bin"
result=$(bash "$R/llama-swap/bin/ai-models")
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
