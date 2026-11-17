# Blender

Pinned Blender with the official Blender Lab MCP add-on and its prepared MCP server, declared in `module.json`. Requires the declared Blender/uv/Python dependencies and an EVA-compatible module runner. A private Xvfb display supports scene screenshots; Windows GUI installation is separate.

```bash
blender -b "$HOME/work/projects/scene/scene.blend" -o //render_ -f 1
blender -b --python "$HOME/work/projects/scene/build.py"
```

Background renders cooperate with the shared GPU lock. MCP-driven GUI rendering does not enforce the same admission path; avoid competing jobs. Save `.blend`, exports and inspected previews under the project rather than relying on agent text.

Install/update verifies the pinned MCP source, interpreter and `mcp-requirements.lock`, then records a prepared executable/provenance in `app/mcp-runtime.json`. Runtime executes `blender-mcp-prepared` directly and never resolves packages. Missing/changed preparation fails; custom owner MCP commands are preserved for review. Agents get the server from the `mcp` declaration in `module.json` (`blender-mcp-prepared --transport stdio`, bridge 127.0.0.1:9876); agent-hermes adds it to Hermes only after the owner approves it. The module never configures Hermes itself.

Claude Code needs a separately approved project entry in `~/work/.mcp.json`; installing the skill does not expose tools. From that trusted project, the owner can use `claude mcp add --scope project --transport stdio blender -- /home/dev/.local/bin/blender-mcp-prepared --transport stdio` after reviewing the Python-execution capability. Preserve other server entries and do not replace a pre-existing `blender` entry without review. Keep Claude's project/server approval prompts enabled; verify discovery with `claude mcp list` and an object-summary request. If approval is outstanding, use the background CLI shown above; it provides saved artifacts, not interactive viewport screenshots.

Windows Codex uses its own MCP configuration, not Claude's `.mcp.json`. Its owner-approved stdio command must cross the existing WSL boundary (`wsl.exe -d <configured-distro> -u dev --exec /home/dev/.local/bin/blender-mcp-prepared --transport stdio`). Reuse the prepared Linux runtime and loopback bridge; do not install a second Blender or auto-copy approval between clients.

Repeat installation currently invokes uv dependency resolution and can need networking even when the prepared runtime already works. An offline update failure is distinct from a broken runtime. Test actual scene/screenshot operations before changing the configuration.

`test.sh` checks Blender startup, add-on and prepared MCP scene/screenshot calls; `--full` also renders on the GPU. MCP initialization failures stop the smoke test, with stderr retained. See [module development](../README.md) for offline regression checks.

The loopback MCP bridge can execute arbitrary Python and has no application login. Do not expose it; other local processes can reach it. Shared-user execution is not adversarial isolation.

`AI_BLENDER_CPU_ONLY=1 blender -b ...` runs a background script that renders nothing (mesh preparation, import/export) without the GPU lock and without unloading the chat model; other modules use this instead of Blender's own executable.


## Local regression tests

Use `python3 -B tests/run.py --module blender --suite fast` from the repository root; use `--suite integration` explicitly for disposable multi-component fixtures. See the [shared test policy](../tests/README.md) for categories, prerequisites and skips. The module-root `test.sh` remains an explicit installer health check, outside the default runner.
