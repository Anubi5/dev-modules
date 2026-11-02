# blender (3D)
Blender (pinned version, checked against blender.org's sha256 list) plus the **official Blender Lab MCP add-on and
server** (pinned v1.0.3) and the **official Hermes Blender plugin**. Agents work in Blender through those MCP tools -
the same tools work for any MCP-capable agent - instead of through scripts written by us.

- A private virtual display (`blender-display`, Xvfb `:99`) gives Blender a real window inside the sandbox, so the
  agent can take screenshots and viewport renders and *see* its scene. Nothing appears on Windows.
- `blender -b ...` still works for scripted renders (takes the shared GPU lock, like `ai gen`).
- `test.sh`: Blender starts, the add-on works, and the MCP path works end to end (scene read + screenshot);
  `--full` also renders on the GPU.
- Open results with Windows Blender (`dev files outputs` / `dev share`).
- Security: the MCP tools run any Python inside Blender, and the bridge (127.0.0.1:9876) has no password. That is fine
  here because the whole sandbox is the boundary (no Windows drives, firewall), but don't expose the port.
