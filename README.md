# Modules

Everything the workstation can do beyond its core is a **module**: one folder here, installed and removed with
`dev mod add <name>` / `dev mod remove <name>`. The core (`ai-sandbox/`) never names a module.
Modules are installed from this git repository (settings `MODULES_REPO`, `MODULES_PATH`, `MODULES_BRANCH`
in `ai-sandbox/config/sandbox.env`).

## A module folder
```
<name>/
  README.md      what it is, how to use it
  install.sh     install or update - runs as dev, safe to run again (skips finished work)
  remove.sh      module-specific cleanup (the runner stops its services and deletes its folder)
  test.sh        quick real check; "test.sh --full" for the slow one
  *.service      optional: systemd --user units, started on demand (ai start <name>)
  bin/           optional: commands linked into ~/.local/bin; bin/ai-<x> becomes "ai <x>"
  skills/<s>/    optional: SKILL.md for agents, installed for Hermes, Claude Code and Codex
```

## Rules
1. A module writes only into its own folder (`~/.local/share/modules/<name>/`) and results into `~/work`.
2. A module never edits the core or another module.
3. A module has a test.

Scripts get these variables: `MOD` (the module's folder), `MOD_NAME`, `WORK_DIR`, `MODELS_MOUNT` (the read-only
Windows models folder), `LLM_PORT` (the model-serving contract: OpenAI-style API + `POST /api/models/unload`),
`CUDA_ARCH`, `MOE_CPU`.

## Adding a model or a ComfyUI node
That is not a new module: add a line to the module's list (`llama-swap/models.json`, `comfyui/models.txt`,
`comfyui/nodes.txt`), push, then `dev mod update <module>` - only what is missing is fetched.

## New module
Copy `_template/`, fill it in (or ask Claude/Codex to, from the tool's README), push, `dev mod add <name>`.
To try a module before pushing: `ai mod add ~/work/projects/<name>` inside the distro.
