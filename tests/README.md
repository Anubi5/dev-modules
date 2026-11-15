# Local test policy

Run from the repository root with an existing Python installation:

```sh
python3 -B tests/run.py --module comfyui --suite fast
python3 -B tests/run.py --module agent-hermes --suite integration
python3 -B tests/run.py --suite fast --list
python3 -B tests/run.py --suite fast --match test_option_limits.py
```

`--module` accepts any of the seven module names, `core`, or `all` (the default).
Repeat it or comma-separate names. `--collect` imports isolated unittest modules
without running cases. `--report path.json` saves results, timings, discovered
identities and explicit skip reasons. Windows can select an existing native
PowerShell with `--powershell path/to/pwsh.exe`. The runner never starts WSL.
Run Linux entries from an already selected Linux environment. Missing tools or
packages produce SKIP; skipped cases produce PARTIAL, never an unconditional PASS.

Every owner uses these populated categories:

| Directory | Meaning | Automatic selection |
|---|---|---|
| `tests/unit/` | Pure logic or isolated mocked collaborators | `fast` |
| `tests/contract/` | Schemas, CLI, capability, safety and graph contracts | `fast` |
| `tests/integration/` | Disposable multi-component fixtures, including fake installers/services | Explicit `integration` |
| `tests/acceptance/` | Installed runtimes, services, model execution and owner effects | Never selected by the runner |
| `tests/fixtures/` | Shared test data, baselines and helpers | Never runnable discovery |

Only create a category when it has content. Cross-module checks use repository
`tests/`; module-owned checks use `<module>/tests/`. The published
`<module>/tests/commands.py` helper stays in place. `inventory.json` records original
names, current locations, ownership, classification and prerequisites. Register
new tests there: an unregistered test or changed unittest identity fails discovery.

Each test runs in a fresh process with disposable HOME, XDG directories and model
paths. This avoids owner configuration/weights leaking into fixture assertions.
The fast suite does not run installed-module health checks, resolve dependencies,
download assets, invoke GPU services or change networking. No framework or package
installation is performed. Ordinary commits use the much smaller staged pre-commit
hook, not the whole fast suite; bug fixes run their relevant tests.

## Installed health and real acceptance

All seven module-root `test.sh` entry points remain because the installer invokes
them. They are explicit installed-module health checks, not fast-suite aliases.
Set `MOD` to the installed module before invoking them. Their effects are:

| Module | Prerequisites and effects |
|---|---|
| agent-hermes | Prepared pinned Hermes and owner profile; quick checks source/profile without inference. `tests/acceptance/hermes_test.sh` loads local inference, invokes tools and replaces its marker file; requires Internet OFF. |
| blender | Installed Blender/add-on/prepared MCP. Quick check can start display/GUI and MCP; full check renders on GPU. |
| character | Python and shipped contracts; quick health runs the relocated isolated tests and CLI help. Real pipeline/Unreal acceptance remains explicitly owner-operated. |
| comfyui | Prepared app, nodes and existing model store. Quick health starts ComfyUI and inspects nodes/models; full `tests/acceptance/comfy_test.py` generates assets. `memtest.py` also runs real GPU work. |
| llama-swap | Prepared server and existing model. Quick health starts the service, performs inference and unloads; full bench runs more models. |
| reverse-tools | Installed command links; prepared optional Java/Ghidra/.NET engines when present. Health starts their diagnostics; it does not install them. |
| rig | Python and shipped contracts for quick health; full health requires installed Blender/UniRig, existing weights and explicit `RIG_TEST_INPUT`, and runs GPU work. A missing fixture remains SKIP. |

STUB and SKIP output is evidence of omission, not real acceptance. Installation,
network policy and GPU acceptance still require their own authorization. Keep
historical failed runs and distinguish source fixtures from installed PC results.
