#!/usr/bin/env bash
## @brief Quick real check (exit 0 = works); --full for the slow one. CI rejects a test that only says "true".
set -euo pipefail
[ "$(python3 -B "$MOD/bin/ai-example" hello)" = 'hello from the template' ]
