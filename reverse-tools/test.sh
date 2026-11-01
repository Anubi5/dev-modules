#!/usr/bin/env bash
## @brief The skill and the rea command are in place; if REA is installed, "rea doctor" must pass.
test -s "$MOD/skills/reverse-tools/SKILL.md" || { echo "skill missing"; exit 1; }
command -v rea >/dev/null || { echo "rea command not linked"; exit 1; }
[ -x "$MOD/app/npm/node_modules/.bin/rea" ] || { echo "  REA not installed yet (installed on first use)"; exit 0; }
rea doctor >/dev/null 2>&1 || { echo "rea doctor fails"; exit 1; }
