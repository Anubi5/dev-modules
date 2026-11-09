# agent-hermes

The local Hermes agent: pinned source, its own locked dependencies and the full reviewed local profile.

Use: `dev hermes` (Windows) or `ai hermes` (inside); both need `dev net off` and a passing profile check.
Needs: internet only during `dev mod add/update agent-hermes`; no GPU of its own (it talks to llama-swap).

- **Install/update:** if `~/.hermes/hermes-agent` already is the pinned, clean commit, it is adopted as-is
  (memory, sessions and config untouched). Otherwise Hermes' **official installer from that exact commit**
  (SHA-256 checked) installs it with `--commit <pin> --non-interactive --skip-browser --skip-computer-use`.
  An existing but different or edited source is never touched: the install stops and says so.
- Then the core's single version check and locked dependency install (`hermes_runtime.py install`), the full
  profile (`hermes_local_setup.py --apply`: local endpoint, agent model, toolsets incl. Blender, memory with
  approved writes, no automatic profile collection) and the profile check.
- The pinned commit lives in the core's `hermes_runtime.py` (one place); this module pins the matching
  installer hash. Changing one without the other fails the install on purpose.
- **Remove:** `dev mod remove agent-hermes` leaves `~/.hermes` (source, memory, sessions) in place.
- Install `agent-hermes` before `blender`: Blender adds the official Hermes Blender plugin.
