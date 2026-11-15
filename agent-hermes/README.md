# agent-hermes

The local Hermes agent: pinned source, its own locked dependencies and the selected local profile.

Use: `dev hermes` (Windows) or `ai hermes` (inside); the Windows entry switches OFF first; Linux entry requires OFF and a passing profile check.
Needs: internet only during `dev mod add/update agent-hermes`; no GPU of its own (it talks to llama-swap).

- **Module tools:** other modules declare MCP servers and skills in their `module.json`; installing a module does not
  expose it. At an attended `dev hermes` start, `integrations.py` reads `ai mod capabilities --json` and asks once
  about every new or changed declaration (its effects are shown), remembers the answer in `~/.hermes/eva-approved.json`
  (kept by `dev state save`), writes approved MCP servers into Hermes' config and copies approved skills to
  `~/.hermes/skills/modules`. Removed or changed providers are dropped at the next start; installs and unattended starts
  never ask and expose nothing new. The profile check refuses any MCP server that is not approved. The file is a
  remembered decision, not a security boundary: Hermes runs as the same user, behind internet OFF.
  Legacy plugin IDs and linked skill markers are retained as history, but require a fresh attended decision
  before authorizing a declared integration. Skills bind approval to their contents. Reconciliation preserves
  unrelated plugins, MCP servers, CLI toolsets and skills; the profile checker can refuse their execution without
  deleting them. An unowned or owner-modified name collision stops reconciliation for owner resolution. Only
  unchanged entries recorded in `eva-approved.json` are removed. A private journal rolls back partial writes on
  failure or the next start; if recovery fails, launch stays blocked and the journal is retained.
- **Install/update:** if `~/.hermes/hermes-agent` already is the pinned, clean commit, it is adopted as-is
  (memory, sessions and config untouched). Otherwise Hermes' **official installer from that exact commit**
  (SHA-256 checked) installs it with `--commit <pin> --non-interactive --skip-browser --skip-computer-use`.
  An existing but different or edited source is never touched: the install stops and says so.
- Then the module's version check and locked dependency install (`hermes_runtime.py install`), the selected
  profile (`hermes_local_setup.py --apply`: local endpoint, agent model, toolsets incl. Blender, memory with
  approved writes, no automatic profile collection) and the profile check.
- The pinned commit lives in this module's `hermes_runtime.py` (one place), next to the matching installer
  hash in `install.sh`. Changing one without the other fails the install on purpose.
- **Self-contained:** everything Hermes-specific is here: pin, profile (`hermes_local_setup.py`,
  `hermes_profile_check.py`, `hermes-local-SOUL.md`), `launch.sh`, the `hermes-dashboard` user service,
  the tool test (`hermes_test.sh`), and the evaluation and acceptance harnesses (`tests/`, e.g. `tests/hermes_eval.sh`).
  The platform only provides the agent gate: `ai hermes` / `ai dashboard` require internet OFF and a failing
  reachability probe, then call `launch.sh` / start the service. The gate is a launch check, not confinement.
- **Your data:** `state.txt` declares what `dev state save` keeps outside the distro (memory, sessions, SOUL);
  the profile itself is re-applied by this module, so `config.yaml` is not carried over.
- **Remove:** `dev mod remove agent-hermes` leaves `~/.hermes` (source, memory, sessions) in place.
- Install `agent-hermes` before `blender`: Blender adds the official Hermes Blender plugin.
