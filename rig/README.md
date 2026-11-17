# Rig

Experimental humanoid rigging with [UniRig](https://github.com/VAST-AI-Research/UniRig) and Blender, pinned assets, a separate environment and a cooperative GPU lease. **F6 visual quality is OPEN ? DEFERRED**: numerical reference acceptance does not establish shoulder/sleeve deformation quality or generalization to other characters.

Requires Blender, uv, Git, Python 3.11 and a CUDA 12.1-compatible GPU. Install through the [module runner](../README.md). Review input geometry and keep originals. Runtime verifies local assets and uses offline Hugging Face/Transformers settings; installation may need approved public downloads.

```bash
ai rig "$HOME/work/inputs/model.glb" --name character --height 1.7 --seed 12345
ai rig "$HOME/work/inputs/model.glb" --facing=+x --triangles 100000 --smooth-iterations 256
ai rig --help
bash rig/test.sh
RIG_TEST_INPUT=/absolute/path/to/reference.glb bash rig/test.sh --full
```

Without `RIG_TEST_INPUT`, full tests explicitly skip real GPU acceptance; that skip is not a quality pass. A supplied missing/empty fixture fails. Defaults live in `profile.json`; avoid copying numerical settings as universal recommendations.

## Contract and results

`contract.json` owns bone names, root/pelvis hierarchy, consuming units, normalized influences, UV/material retention, triangle and deformation/render gates. `models.json` pins source/checkpoints. Preprocessing records topology, retention and sampled surface drift; it does not establish game readiness.

Results under `$WORK_DIR/outputs/rig/<name>-<job-id>/` include FBX, GLB, Blender source, renders and `record.json`. `RIG_WORK_DIR` selects an isolated work directory. `--json` emits `local-ai.producer/1`; progress goes to stderr. Exit codes are 0 success, 1 runtime/input failure, 2 validation failure, 130 interrupted. Character pipelines consume FBX-only rig manifests; other formats retain their own metadata in the producer record.

Orientation checks the observed four-joint toe-ended UniRig leg chains immediately after skeleton generation and again during finish. Declare input facing correctly; the output normalizes to -Y. Ambiguous/heel-ended rigs and mirrored/singular transforms are refused. A reflection baked into geometry cannot be inferred from transform determinants alone. Independent anatomical/motion review is still required.

Source integrity checks reject changed/missing/extra files, symlinks, ignored bytecode and submodules before importing the pinned tree. Preserve unfamiliar content for review rather than auto-deleting it. This is not immutable storage or protection against concurrent privileged tampering.

FBX import compatibility must be tested in the consumer; an FBX header alone is insufficient. Procedural poses and independent re-imports check selected numerical/render properties. Unreal animation validation is a separate stage marked `not_run`; neither stage establishes visual quality automatically.

## Local walk preview

`ai animate <model_named.glb> --motion walk --duration 2` creates a bounded 24 fps Cycles preview with the existing Blender and ffmpeg/libx264. Duration is 1-10 seconds; start with two seconds before a longer render. The adjacent `verify_glb.json` must contain passing finish/render checks and the exact input hash (rerun the current `verify.py` for older reports). `--validation` selects another report; `--output` requires a new directory. The command saves `scene.blend`, `motion.glb`, `preview.mp4`, frames, logs and a provenance `record.json`; inspect the clip before claiming motion quality. Walk is deterministic and experimental, not mocap or a foot-contact guarantee. Unreal handoff remains available.

## Interruption and recovery

Completed prepared mesh, skeleton, skin and merged GLB are copied to the job output's `not_validated/` directory. `record.json` records their hashes alongside input, module and model identity; `--json` exposes them separately as `intermediates`. They are not validated deliverables, including after a failed finish. Automatic stage resume is not implemented.

The lease stops/verifies chat before inference and restores it only if previously active; stop/restart failures are errors. This coordinates cooperating clients, not arbitrary Windows/browser GPU work. Normal cancellation terminates the owned inference group and releases its lease.

SIGKILL or host loss can leave children holding the lease and stale state. Inspect retained records and current process identities, establish that owned GPU work has ended, then restore chat only when the initial state requires it. Do not delete lock files, signal historical PIDs blindly or claim automatic stale-job/native crash recovery. Use the platform's attended Unreal handoff for native animation work.


## Local regression tests

Use `python3 -B tests/run.py --module rig --suite fast` from the repository root; use `--suite integration` explicitly for disposable multi-component fixtures. See the [shared test policy](../tests/README.md) for categories, prerequisites and skips. The module-root `test.sh` remains an explicit installer health check, outside the default runner.
