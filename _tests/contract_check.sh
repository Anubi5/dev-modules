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
[ $bad = 0 ] && echo "  module contract checks: OK"; exit $bad
