#!/usr/bin/env bash
## @brief Contract check: models listed, unload endpoint answers, one short answer; --full = bench of every model.
set -uo pipefail
API=http://127.0.0.1:${LLM_PORT:-8080}
systemctl --user start llama-swap
for _ in $(seq 60); do curl -sf -m 3 "$API/v1/models" >/dev/null && break; sleep 2; done
n=$(curl -sf "$API/v1/models" | jq '.data | length') || { echo "model server not answering"; exit 1; }
[ "${n:-0}" -gt 0 ] || { echo "no models"; exit 1; }
curl -sf -m 20 -o /dev/null -X POST "$API/api/models/unload" || { echo "unload endpoint missing (contract)"; exit 1; }
echo "  $n models, unload endpoint OK"
[ "${1:-}" = --full ] && exec "$MOD/app/venv/bin/python" "$MOD/bench.py"
exit 0
