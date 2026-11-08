# Modules

Everything the workstation can do beyond its core is a **module**: one folder here, installed and removed with
`dev mod add <name>` / `dev mod remove <name>`. The core only names modules as defaults (`$DefaultModules`) and in shortcuts such as `dev chat` / `dev comfy`. Try a local module first with `ai mod add --from <folder>`.
Modules are installed from this git repository (settings `MODULES_REPO`, `MODULES_PATH`, `MODULES_BRANCH`
in `ai-sandbox/config/sandbox.env`).

## A module folder
```
<name>/
  README.md      what it is, how to use it
  install.sh     install or update - runs as dev, safe to run again (skips finished work)
  remove.sh      optional module-specific cleanup (if it fails, nothing is removed)
  test.sh        required: quick check; "test.sh --full" for the slow one
  *.service      optional: systemd --user units, started on demand (ai start <name>)
  bin/           optional: commands linked into ~/.local/bin; bin/ai-<x> becomes "ai <x>"
  skills/<s>/    optional: SKILL.md for agents, installed for Hermes, Claude Code and Codex
```

## Rules
1. A module writes only into its own folder (`~/.local/share/modules/<name>/`) and results into `~/work`.
2. A module never edits the core or another module.
3. A module has a test.

Scripts get these variables: `MOD` (the module's folder), `MOD_NAME`, `WORK_DIR`, `MODELS_MOUNT` (the read-only
Windows models folder), `MODELS_RW` (the same folder writable - only mounted while Windows runs `dev mod add|update`;
model files go there, never into the module folder), `LLM_PORT` (the model-serving contract: OpenAI-style API + `POST /api/models/unload`),
`CUDA_ARCH`, `MOE_CPU`.

## Adding a model or a ComfyUI node
That is not a new module: add a line to the module's list (`llama-swap/models.json`, `comfyui/models.txt`,
`comfyui/nodes.txt`), push, then `dev mod update <module>`. The target source is
fetched first; missing model files require a reviewed pin, byte size and licence,
and exact Windows DOWNLOAD approval. Ordinary updates and unattended runs do not
receive model approval.

## Module-owned download plans

A module that manages model files supplies `model-plan.sh`, which invokes its own
fetcher with `--plan --json`. The hook emits `{ "schema": 1, "files": [...] }`;
each missing file reports path, bytes, source, licence, sha256, eligible and an
optional reason. Planning reads declarations only: no downloads, model execution,
state/configuration writes or installs. A module without managed files needs no
hook. A new module requires no core parser change.

The core fetches the configured module revision, retains a clean detached source
checkout, aggregates hook results and returns the exact source revision. Windows
asks once and passes approved SHA-256 arguments plus that revision. Execution
uses the same checkout even if another fetch advances the source cache. Existing
code/data are preserved; retained source checkouts are not automatically deleted.

## New module
Copy `_template/`, fill it in (or ask Claude/Codex to, from the tool's README), push, `dev mod add <name>`.
To try a module before pushing: `ai mod add ~/work/projects/<name>` inside the distro.
