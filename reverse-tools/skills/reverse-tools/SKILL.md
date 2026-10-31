---
name: reverse-tools
description: Install and use REA (reverse-engineering MCP + CLI) with Ghidra and other free tools, in your own folders without root, when the user wants to inspect, decompile or learn reverse engineering of a file.
---

# Reverse-engineering tools (on request)

## When to Use
The user asks to inspect, decompile or understand a binary/app (their own or licensed software, learning, mods).
Never use it to remove licence checks, copy protection or anti-cheat.

## Install (once, only when first needed; everything under ~/.local, no sudo exists)
1. Check first: `command -v rea && rea doctor` - if it passes, skip to Use.
2. REA: `npm config set prefix ~/.local && npm install --global rea-agents`
3. JDK 21 (portable): download `https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse`
   and unpack to `~/.local/opt/jdk-21`.
4. Ghidra: REA needs the exact release named in REA's docs (`rea doctor` says which). Get the official
   `ghidra_<version>_PUBLIC_*.zip` from github.com/NationalSecurityAgency/ghidra releases and unpack to `~/.local/opt/`.
5. Optional extras as needed: jadx (github.com/skylot/jadx releases, zip), rizin (official static Linux x86_64 release).
6. Write `~/.config/rea/env` with `export GHIDRA_INSTALL_DIR=...` and `export JAVA_HOME=~/.local/opt/jdk-21`,
   and source it in `~/.bashrc`. Then `rea doctor` must pass.
7. Registering REA with agents (`rea setup`) changes agent configs: tell the user to run it themselves in `dev shell`.

## Use
- Work in `~/work/projects/<name>/`: copy the target to `original/` (never modify it), results in `output/`, notes in `README.md`.
- Prefer static analysis (reads files only). Before anything that RUNS the target, tell the user and ask them to run
  `dev net off` first.
- Report findings with the evidence REA returns; say plainly what is uncertain.
