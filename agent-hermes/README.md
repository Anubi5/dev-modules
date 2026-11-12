# agent-hermes

The local Hermes agent: pinned source, its own locked dependencies and the selected local profile.

Use: `dev hermes` (Windows) or `ai hermes` (inside); the Windows entry switches OFF first; Linux entry requires OFF and a passing profile check.
Needs: internet only during `dev mod add/update agent-hermes`; no GPU of its own (it talks to llama-swap).

- **Install/update:** if `~/.hermes/hermes-agent` already is the pinned, clean commit, it is adopted as-is
  (memory, sessions and config untouched). Otherwise Hermes' **official installer from that exact commit**
  (SHA-256 checked) installs it with `--commit <pin> --non-interactive --skip-browser --skip-computer-use`.
  An existing but different or edited source is never touched: the install stops and says so.
- Then the core's single version check and locked dependency install (`hermes_runtime.py install`), the selected
  profile (`hermes_local_setup.py --apply`: local endpoint, agent model, toolsets incl. Blender, memory with
  approved writes, no automatic profile collection) and the profile check.
- The pinned commit lives in the core's `hermes_runtime.py` (one place); this module pins the matching
  installer hash. Changing one without the other fails the install on purpose.
- **Your data:** `state.txt` declares what `dev state save` keeps outside the distro (memory, sessions, SOUL);
  the profile itself is re-applied by this module, so `config.yaml` is not carried over.
- **Remove:** `dev mod remove agent-hermes` leaves `~/.hermes` (source, memory, sessions) in place.
- Install `agent-hermes` before `blender`: Blender adds the official Hermes Blender plugin.
