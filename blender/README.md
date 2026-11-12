# Blender

Pinned Blender with the official Blender Lab MCP add-on/server and Hermes plugin. Requires the declared Blender/uv/Python dependencies and an EVA-compatible module runner. A private Xvfb display supports scene screenshots; Windows GUI installation is separate.

```bash
blender -b "$HOME/work/projects/scene/scene.blend" -o //render_ -f 1
blender -b --python "$HOME/work/projects/scene/build.py"
```

Background renders cooperate with the shared GPU lock. MCP-driven GUI rendering does not enforce the same admission path; avoid competing jobs. Save `.blend`, exports and inspected previews under the project rather than relying on agent text.

Install/update verifies the pinned MCP source, interpreter and `mcp-requirements.lock`, then records a prepared executable/provenance in `app/mcp-runtime.json`. Runtime executes `blender-mcp-prepared` directly and never resolves packages. Missing/changed preparation fails; custom owner MCP commands are preserved for review. The Hermes plugin uses a bare linked launcher name rather than an absolute command path.

Repeat installation currently invokes uv dependency resolution and can need networking even when the prepared runtime already works. An offline update failure is distinct from a broken runtime. Hermes can warn `Unknown toolsets: blender` despite successful MCP calls; test actual scene/screenshot operations before changing the configuration.

`test.sh` checks Blender startup, add-on and prepared MCP scene/screenshot calls; `--full` also renders on the GPU. MCP initialization failures stop the smoke test, with stderr retained. See [module development](../README.md) for offline regression checks.

The loopback MCP bridge can execute arbitrary Python and has no application login. Do not expose it; other local processes can reach it. Shared-user execution is not adversarial isolation.
