#!/usr/bin/env bash
## @brief Contract check: models listed, one real answer, unload empties /running; --full = bench of every model.
set -uo pipefail
API=http://127.0.0.1:${LLM_PORT:-8080}
systemctl --user start llama-swap
for _ in $(seq 60); do curl -sf -m 3 "$API/v1/models" >/dev/null && break; sleep 2; done
n=$(curl -sf "$API/v1/models" | jq '.data | length') || { echo "model server not answering"; exit 1; }
[ "${n:-0}" -gt 0 ] || { echo "no models"; exit 1; }
# one real answer from the smallest model (loads it), then unload and check nothing stays loaded
m=$(curl -sf "$API/v1/models" | jq -r '[.data[].id] | (map(select(test("coder"))) + .)[0]')
a=$(curl -sf -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"max_tokens\":8,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}]}" \
    | jq -r '.choices[0].message | (.content // "") + (.reasoning_content // "")')
[ -n "$a" ] || { echo "no answer from $m (inference broken)"; exit 1; }
curl -sf -m 20 -o /dev/null -X POST "$API/api/models/unload" || { echo "unload endpoint missing (contract)"; exit 1; }
for _ in $(seq 15); do [ "$(curl -sf "$API/running" | jq '.running | length' 2>/dev/null)" = 0 ] && break; sleep 2; done
[ "$(curl -sf "$API/running" | jq '.running | length' 2>/dev/null)" = 0 ] || { echo "model still loaded after unload"; exit 1; }
echo "  $n models; $m answered ($(echo "$a" | head -c 20 | tr -d '\n')); unload frees it"
[ "${1:-}" = --full ] && exec "$MOD/app/venv/bin/python" "$MOD/bench.py"
exit 0
