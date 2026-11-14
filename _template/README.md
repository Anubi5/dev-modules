# <name>
What it does, in one sentence. Upstream: <link>.

Use: `ai example hello` ...  Results go to `~/work/outputs/<name>/`.
Needs: GPU no/yes (through the shared `/tmp/ai-gpu.lock`) · internet at install / at runtime · port (if a service).

## Anatomy (leave out what a small module does not need)

```
module.json     identity, version, capabilities, requires/optional, mcp, effects, skills, ui (manifest v2)
install.sh  test.sh  remove.sh
state.txt       user data dev state save keeps across rebuilds (never weights, programs or caches)
bin/            thin entry points; their parsers define the arguments and answer --help
operations/     one file per user operation (larger modules only)
families/       per model family: package.json (+ family.py only where needed)
features/       optional features; per-family files inside a feature folder when they differ
common/         mechanics shared inside this module only
skills/<name>/  SKILL.md (+ references/, scripts/, assets/) per agentskills.io
tests/          commands.py (subcommands and read-only fixtures for the effects contract), unit tests
```

`module.json` declares every command in `bin/` (or `<command> <subcommand>`) under `effects`: `read_only`,
`writes[:scope]`, `gpu`, `network`, `consequential`. A command without an entry is not authorised. Effects describe;
they never grant privileges. Other modules are used only through their commands and JSON contracts, never their
folders; integrations are declared here and pulled by the consumer.
