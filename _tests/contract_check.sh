#!/usr/bin/env bash
## @file contract_check.sh
## @brief Module rules that can be checked without a GPU: every module's test.sh really checks something (not just
##        "true"), and model files in formats that can run code when loaded (.bin .pt .pth .ckpt .pkl) carry a pinned
##        sha256 in comfyui/models.txt. @param $1 repo root (default: this repo) @return 0 when every rule holds
R=${1:-$(cd "$(dirname "$0")/.." && pwd)}; bad=0
for d in "$R"/*/; do
  n=$(basename "$d"); case "$n" in _*|.*) continue ;; esac
  [ -f "$d/test.sh" ] || continue                                  # missing files are reported by the presence check
  real=$(sed -E 's/\s+#.*$//' "$d/test.sh" | grep -vE '^\s*(#|$)|^\s*(set( +-?[a-z]+)+|true|:|exit( 0)?)\s*;?\s*$' | head -1)
  [ -n "$real" ] || { echo "  $n/test.sh does not test anything (only comments, set, true or exit 0)"; bad=1; }
done
if [ -f "$R/comfyui/models.txt" ]; then
  while read -r line; do
    echo "  comfyui/models.txt: no sha256 for a file that can run code when loaded: $line"; bad=1
  done < <(grep -vE '^\s*(#|$)' "$R/comfyui/models.txt" | grep -iE '\.(bin|pt|pth|ckpt|pkl)(\s|$)' | grep -v 'sha256:')
fi
# Optional module.json: only title/url/service, strings; url a loopback page; service one of the module's own units.
for m in "$R"/*/module.json; do
  [ -f "$m" ] || continue; n=$(basename "$(dirname "$m")")
  python3 -B - "$m" "$R/$n" <<'PY' || bad=1
import json, re, sys
from pathlib import Path
path, module = sys.argv[1], Path(sys.argv[2])
try: data = json.loads(Path(path).read_text())
except ValueError as error: sys.exit(f'  {module.name}/module.json: {error}')
problems = [f'unknown key {k}' for k in data if k not in ('title', 'url', 'service')]
problems += [f'{k} is not a string' for k, v in data.items() if not isinstance(v, str)]
if isinstance(data.get('url'), str) and not re.match(r'^http://127\.0\.0\.1:(\d+|\$\{[A-Z_][A-Z0-9_]*\})/', data['url']): problems.append('url is not a 127.0.0.1 page')
if isinstance(data.get('service'), str) and not (module / (data['service'] + '.service')).is_file(): problems.append('service is not a unit of this module')
for p in problems: print(f'  {module.name}/module.json: {p}')
sys.exit(1 if problems else 0)
PY
done
[ $bad = 0 ] && echo "  module contract checks: OK"; exit $bad
