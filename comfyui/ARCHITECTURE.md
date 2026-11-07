# Generation architecture

`gen.py` remains the command-line entry point and the single owner of execution:
GPU leases, services, cancellation, jobs, retries and output records. Existing
commands, options, functions used by integrations, limits and exit codes remain
compatible. Repository updates do not install into the workstation automatically.

The first extraction separates workflow construction from that runtime:

| Component | Responsibility |
| --- | --- |
| `generation/graph.py` | Recipe binding, switch pruning, graph links and LoRA insertion. |
| `generation/models.py` | Shared declarative profile catalogue and style defaults. |
| `generation/image.py` | Narrow builders for differing image/editing architectures and pose graphs. |
| `generation/video.py` | Video graph construction. |
| `generation/mesh.py` | GLB inspection, quality presets and 3D validation verdicts. |

These components import without the CLI, service runtime, model assets or PyTorch.
They construct or inspect data; they never acquire GPU ownership or execute jobs.
Profiles remain in `profiles.json` and recipes in `recipes/`; the catalogue has one
shared owner, with per-request profile copies. `gen.py` re-exports existing names
for compatibility while internal builders call their own component functions.

`generation.models.package(name, kind)` resolves a version-1 declarative model
package into a capability, architecture and detached settings. Existing image/3D
profiles retain their namespace; `video_models` holds video packages and
`default_video` chooses the default. A Wan package declares its model/encoder/VAE,
sampling settings, dimensions and frame rate. Its adapter owns node wiring and
rejects a different architecture; metadata cannot name Python entry points.
Adding a compatible video model requires data and the existing model-install
manifest, not another adapter or dependency environment. `ai gen video ... --model
NAME` selects such a package; the old no-model command retains its exact graph.

Run the focused offline checks with:

```bash
python3 -B -m unittest discover -s comfyui -p test_generation.py
python3 -B comfyui/check_workflows.py
```

`tests/workflow-baseline.json` records canonical SHA-256 hashes of all 233 graphs
from the validated pre-extraction revision. A structural refactor must match them
exactly. Update that baseline only for an intentional, separately reviewed workflow
behavior change, with the corresponding schema and capability tests.

This is the first increment, not the final runtime split. Subsequent increments
will define model-package interfaces and file-based mesh/rig/animation contracts,
then productize the Unreal hand-off and composable character orchestration.
Keep adapters specific to architectural differences, installation dependencies
isolated, and execution ownership explicit. F6 remains OPEN — DEFERRED; rigging
quality and automatic crash recovery are outside this refactor.
