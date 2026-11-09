# blender (3D)
Blender (pinned version, checked against blender.org's sha256 list) plus the **official Blender Lab MCP add-on and
server** (pinned v1.0.3) and the **official Hermes Blender plugin**. Agents work in Blender through those MCP tools -
the same tools work for any MCP-capable agent - instead of through scripts written by us.

- A private virtual display (`blender-display`, Xvfb `:99`) gives Blender a real window inside the sandbox, so the
  agent can take screenshots and viewport renders and *see* its scene. Nothing appears on Windows.
- `blender -b ...` still works for scripted renders (takes the shared GPU lock, like `ai gen`).
- `test.sh`: Blender starts, the add-on works, and the MCP path works end to end (scene read + screenshot);
  `--full` also renders on the GPU. The smoke uses `blender-mcp-prepared`, never
  `uv tool run`; missing preparation fails rather than resolving dependencies.
  Server stderr remains visible, and failed initialization stops the protocol.
- Open results with Windows Blender (`dev files outputs` / `dev share`).
- Install/update records the pinned MCP executable, Git provenance and installed
  dependency metadata in `app/mcp-runtime.json`, and retains the old plugin MCP
  configuration before selecting `blender-mcp-prepared`. Launch executes that
  prepared binary directly; it never calls uv or another installer. Missing or
  changed preparation fails with an explicit `dev mod update blender` instruction.
  Custom owner commands are preserved and require review before replacement.
  Python 3.14.7 and the 38 exact transitive versions in `mcp-requirements.lock`
  are selected during installation; the prepared environment must match that lock.
- Security: the MCP tools run any Python inside Blender, and the bridge (127.0.0.1:9876) has no password. That is fine
  here because the whole sandbox is the boundary (no Windows drives, firewall), but don't expose the port.
