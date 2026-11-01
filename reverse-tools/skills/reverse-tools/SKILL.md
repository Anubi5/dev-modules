---
name: reverse-tools
description: Install and use REA (reverse-engineering MCP + CLI) with Ghidra and other free tools, in your own folders without root, when the user wants to inspect, decompile or learn reverse engineering of a file.
---

# Reverse-engineering tools (on request)

## When to Use
The user asks to inspect, decompile or understand a binary/app (their own or licensed software, learning, mods).
Never use it to remove licence checks, copy protection or anti-cheat.

## Install (once, only when first needed; everything inside the module folder, no sudo exists)
Everything goes under `M=~/.local/share/modules/reverse-tools/app` - so `dev mod remove reverse-tools` removes it all.
Do not change `~/.bashrc`, the global npm prefix or anything outside `$M`. The `rea` command (already linked) uses `$M`.
1. Check first: `command -v rea && rea doctor` - if it passes, skip to Use.
2. REA: `npm install --prefix "$M/npm" rea-agents`
3. JDK 21 (portable): download `https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse`
   and unpack to `$M/jdk-21`.
4. Ghidra: REA needs the exact release named in REA's docs (`rea doctor` says which). Get the official
   `ghidra_<version>_PUBLIC_*.zip` from github.com/NationalSecurityAgency/ghidra releases and unpack to `$M/` (as `$M/ghidra`).
5. Optional extras as needed, also into `$M`: jadx (github.com/skylot/jadx releases, zip), rizin (official static Linux x86_64 release).
6. Run `rea doctor` (the `rea` command sets `JAVA_HOME=$M/jdk-21` and `GHIDRA_INSTALL_DIR=$M/ghidra` itself); it must pass.
7. Registering REA with agents (`rea setup`) changes agent configs: tell the user to run it themselves in `dev shell`.

## Use
- Work in `~/work/projects/<name>/`: copy the target to `original/` (never modify it), results in `output/`, notes in `README.md`.
- Prefer static analysis (reads files only). Before anything that RUNS the target, tell the user and ask them to run
  `dev net off` first.
- Report findings with the evidence REA returns; say plainly what is uncertain.
