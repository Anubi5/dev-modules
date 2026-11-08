---
name: reverse-tools
description: Install and use REA (reverse-engineering MCP + CLI) with Ghidra, ILSpy (.NET/Unity) and other free tools, in your own folders without root, when the user wants to inspect, decompile or learn reverse engineering of a file.
---

# Reverse-engineering tools (on request)

## When to Use
The user asks to inspect, decompile or understand a binary/app (their own or licensed software, learning, mods).
Never use it to remove licence checks, copy protection or anti-cheat.

## Install (once, only when first needed; everything inside the module folder, no sudo exists)
Start with `M="$HOME/.local/share/modules/reverse-tools/app"; mkdir -p "$M"`.
Everything goes there, so `dev mod remove reverse-tools` removes it all.
Do not change `~/.bashrc`, the global npm prefix or anything outside `$M`. The `rea` command (already linked) uses `$M`.
1. Check first: `command -v rea && rea doctor --provider ghidra`. This scopes readiness
   to our Ghidra capability; unrelated agent registrations are not required. A real
   static import must also pass: doctor alone did not catch the Java mismatch.
2. REA: `npm install --prefix "$M/npm" rea-agents@6.8.0` (raise deliberately).
3. Match Java to the pinned Ghidra release. Ghidra 12.1.4 needs **Java 25**;
   Java 21 fails with `UnsupportedClassVersionError` despite passing doctor.
   Get the official Temurin Linux x64 JDK archive and SHA-256 from
   `https://api.adoptium.net/v3/assets/latest/25/hotspot?architecture=x64&image_type=jdk&os=linux`.
   Verify the archive against that checksum, unpack with its top directory removed
   into `$M/jdk-25`, and verify `$M/jdk-25/bin/java -version`. Keep older JDKs until
   the static test passes; never change global Java or shell profiles.
4. Ghidra: REA needs the exact release named in REA's docs (`rea doctor` says which). Get the official
   `ghidra_<version>_PUBLIC_*.zip` from github.com/NationalSecurityAgency/ghidra releases and unpack to `$M/` (as `$M/ghidra`).
5. Optional extras as needed, also into `$M`: jadx (github.com/skylot/jadx releases, zip), rizin (official static Linux x86_64 release).
   ILSpy for .NET programs and Unity games (C# DLLs such as `Assembly-CSharp.dll`; Ghidra handles these poorly):
   Use the official .NET 10 release metadata and verify its SDK archive SHA-512
   before unpacking into `$M/dotnet` (validated SDK: 10.0.403). Set
   `DOTNET_CLI_HOME="$M/dotnet-home"`, `NUGET_PACKAGES="$M/nuget"`,
   `DOTNET_CLI_TELEMETRY_OPTOUT=1`, and `DOTNET_GENERATE_ASPNET_CERTIFICATE=false`.
   Then `DOTNET_ROOT="$M/dotnet" "$M/dotnet/dotnet" tool install ilspycmd --version 11.2.0.10050 --tool-path "$M/ilspy"`.
   Write the installed version (`ilspy --version`) into `$M/VERSIONS.txt`. The `ilspy` command (already linked) uses `$M`.
6. Run `rea doctor --provider ghidra` (the wrapper selects the module's Java 25 when
   present, otherwise its legacy Java 21). Then analyze a read-only copy of an owned
   or licensed test binary: `rea analyze path/to/copy --provider ghidra --format json`.
   Preserve errors; never keep retrying an unchanged failure or register other agents
   merely to clear the global doctor report.
7. Registering REA with agents (`rea setup`) changes agent configs: tell the user to run it themselves in `dev shell`.

## Use
- Work in `~/work/projects/<name>/`: copy the target to `original/` (never modify it), results in `output/`, notes in `README.md`.
- .NET / Unity: `ilspy -p -o output/src original/<file>.dll` writes a C# project you can read; for Unity games the
  game code is in `<Game>_Data/Managed/Assembly-CSharp.dll`.
- Prefer static analysis (reads files only). Before anything that RUNS the target, tell the user and ask them to run
  `dev net off` first.
- Report findings with the evidence REA returns; say plainly what is uncertain.
