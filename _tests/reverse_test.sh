#!/usr/bin/env bash
# Ghidra scope excludes unrelated registrations, but still fails on an unavailable engine or broken ILSpy.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/module/skills/reverse-tools" "$T/module/app/npm/node_modules/.bin" "$T/module/app/ilspy" "$T/bin"
cp "$R/reverse-tools/skills/reverse-tools/SKILL.md" "$T/module/skills/reverse-tools/"
cat > "$T/bin/rea" <<'SH'
#!/usr/bin/env bash
[ "$*" = 'doctor --provider ghidra' ] && [ "${ENGINE_BROKEN:-0}" = 0 ]
SH
cat > "$T/bin/ilspy" <<'SH'
#!/usr/bin/env bash
[ "${ILSPY_BROKEN:-0}" = 0 ]
SH
chmod +x "$T/bin/rea" "$T/bin/ilspy"
cp "$T/bin/rea" "$T/module/app/npm/node_modules/.bin/rea"
cp "$T/bin/ilspy" "$T/module/app/ilspy/ilspycmd"
export PATH="$T/bin:$PATH" MOD="$T/module"
bash "$R/reverse-tools/test.sh"
if ENGINE_BROKEN=1 bash "$R/reverse-tools/test.sh"; then echo 'FAIL broken Ghidra was accepted'; exit 1; fi
if ILSPY_BROKEN=1 bash "$R/reverse-tools/test.sh"; then echo 'FAIL broken ILSpy was accepted'; exit 1; fi
echo 'Reverse-tools scoped readiness and negative cases: OK'
