# Test inventory preservation

Original reviewed source: `1fb50ec68239b0251521040e57fad9fbbb5e9c7e`.

Python unittest identities: **244 -> 252**. Missing: **0**. Eight new ComfyUI regressions account for the increase.

| Owner | Category | Original files / Python cases | Final files / Python cases |
|---|---|---:|---:|
| agent-hermes | unit | 2 / 12 | 2 / 12 |
| agent-hermes | contract | 5 / 31 | 5 / 31 |
| agent-hermes | integration | 3 / 24 | 3 / 24 |
| agent-hermes | acceptance | 5 / 0 | 5 / 0 |
| agent-hermes | fixtures | 1 / 0 | 1 / 0 |
| blender | contract | 1 / 7 | 1 / 7 |
| blender | integration | 1 / 2 | 1 / 2 |
| blender | acceptance | 3 / 0 | 3 / 0 |
| character | contract | 2 / 6 | 2 / 6 |
| character | integration | 1 / 8 | 1 / 8 |
| character | acceptance | 1 / 0 | 1 / 0 |
| comfyui | unit | 4 / 36 | 4 / 36 |
| comfyui | contract | 6 / 47 | 8 / 55 |
| comfyui | integration | 2 / 4 | 2 / 4 |
| comfyui | acceptance | 3 / 0 | 3 / 0 |
| comfyui | fixtures | 2 / 0 | 2 / 0 |
| core | contract | 9 / 24 | 9 / 24 |
| core | integration | 3 / 2 | 3 / 2 |
| core | fixtures | 3 / 0 | 3 / 0 |
| llama-swap | unit | 3 / 13 | 3 / 13 |
| llama-swap | contract | 1 / 0 | 1 / 0 |
| llama-swap | acceptance | 1 / 0 | 1 / 0 |
| reverse-tools | contract | 1 / 0 | 1 / 0 |
| reverse-tools | acceptance | 1 / 0 | 1 / 0 |
| rig | unit | 4 / 16 | 4 / 16 |
| rig | contract | 4 / 7 | 4 / 7 |
| rig | integration | 1 / 5 | 1 / 5 |
| rig | acceptance | 1 / 0 | 1 / 0 |

Files include test-only helpers/data and explicit installed health/acceptance entries; Python cases count unittest methods, not subtests or shell assertions. `inventory.json` retains every original name, current entry point, prerequisites and effects. Acceptance rows are retained but were not run. Categories with no files are omitted, not created empty.
