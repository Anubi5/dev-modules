#!/usr/bin/env bash
## @brief The skill and the rea/ilspy commands are in place; installed tools must start ("rea doctor", "ilspy --version").
test -s "$MOD/skills/reverse-tools/SKILL.md" || { echo "skill missing"; exit 1; }
command -v rea >/dev/null || { echo "rea command not linked"; exit 1; }
command -v ilspy >/dev/null || { echo "ilspy command not linked"; exit 1; }
if [ -x "$MOD/app/ilspy/ilspycmd" ]; then ilspy --version >/dev/null 2>&1 || { echo "ilspy does not start"; exit 1; }
else echo "  ILSpy not installed yet (installed on first use)"; fi
if [ -x "$MOD/app/npm/node_modules/.bin/rea" ]; then rea doctor >/dev/null 2>&1 || { echo "rea doctor fails"; exit 1; }
else echo "  REA not installed yet (installed on first use)"; fi
