# Rig

Local humanoid rigging with [UniRig](https://github.com/VAST-AI-Research/UniRig)
and Blender, with an independent Python environment, pinned checkpoints,
shared GPU lease and a record for every started job. This branch is staged and
tested independently of the installed workstation; it has not been merged.

**Batch 1 source corrections are offline-tested only.** Historical GPU/Unreal
acceptance predates the new chat, orientation, render and source-integrity checks.
Do not treat it as hardware acceptance of this revision or install it in production.

## Commands

Requires the Blender module, `uv`, Git, Python 3.11 and a CUDA 12.1 compatible GPU.
Install through the module manager (`dev mod add rig` once this branch is selected).
The installer verifies the upstream commit, exact dependency lock and public
checkpoint hashes; it can be rerun. No account token is required.

```bash
ai rig model.glb --name character --height 1.7 --seed 12345
ai rig model.glb --triangles 100000 --smooth-iterations 256
ai rig model.glb --facing=+x     # Declare Blender-space input facing; normalize to -Y
bash test.sh                    # Offline contract mutation checks and CLI help
RIG_TEST_INPUT=model.glb bash test.sh --full   # Full GPU inference and exports
```

Results are under `$WORK_DIR/outputs/rig/<name>-<job-id>/`: named and scaled
`model_unreal.fbx`, `model_named.glb`, `model_rig.blend`, renders, logs and
`record.json`. `RIG_WORK_DIR` overrides the core work directory for isolated
acceptance runs. Temporary files are retained under that work directory.
Exit codes: 0 success, 1 runtime/input failure, 2 validation failure, 130 interrupted.
Cancellation terminates the inference process group, releases the GPU lease and
restores chat if it was previously active. Inference uses verified local assets
with Hugging Face/Transformers offline; installation may download pinned public files.

While holding the lease, rigging stops and verifies llama-swap before inference,
then verifies restoration if it was active. Failed stops/restarts are errors.
This coordinates cooperating WSL clients; it does not prevent a person from
manually restarting a service or an unrelated Windows program using the GPU.
SIGKILL/power loss cannot run restoration: inspect the retained job, confirm its
children have stopped and run `ai start llama-swap` if restoration is needed.
Automated stale-job recovery and crash/contended-GPU acceptance remain outstanding.

`contract.json` specifies required bone names, an unweighted root above pelvis,
centimetre units, four normalized influences, UV/material retention, triangle
budgets, and deformation/render gates. Creature rigs need a separate contract.
`models.json` pins the upstream revision and both checkpoint SHA-256 digests.

## Orientation and integrity limits

The supported orientation reference is **UniRig's heel-ended reference chain**,
not a general anatomical classifier. Default declared input facing remains `-y`.
Both terminal heel segments and hip separation must agree at the existing .9
alignment threshold; rotated inputs need the correct `--facing=...` declaration.
The output is normalized to -Y and semantic hand/thigh sides are checked again
after re-import. Unknown/toe-ended rigs, ambiguous evidence, and mirrored/singular
object transforms fail. A reflection baked into geometry cannot be identified
from transform determinants alone; semantic/anatomical correctness still needs
the declared fixture and real animation review.

Before importing upstream code, the command checks all files against the pinned
Git tree, refusing extra files/directories (including ignored Python/bytecode),
symlinks, submodules, missing files and changed/staged bytes. It records the checked
tree digest. Inspect an integrity error and re-provision a clean staging source;
do not delete unknown files automatically. This preflight is not an immutable
mount and does not defend against concurrent privileged source/runtime tampering.

The FBX reference is Unreal's **FBX SDK 2020.2** import pipeline. That SDK release
is distinct from an FBX file's binary version: Blender's intermediate is 7.4.
Unreal import and animated round-trip acceptance are mandatory; a file header
alone does not establish compatibility.

Preprocessing uses Blender's Decimate modifier before inference. Candidate budgets
were 100k and 50k triangles. The 100k profile with 256 smoothing iterations passed
the reference Unreal walk/jog test; 50k exceeded the deformation limit and was
rejected. This is a tested reference profile, not a guarantee for arbitrary meshes.
`prepare_mesh.py` records topology, UV retention,
input/output hashes and sampled surface drift. It does not claim game readiness.

The command checks procedural poses, rendered images and independent FBX/GLB
re-imports (height, bones, root, weights, UVs and triangle limit). Unreal animation
validation is a separate acceptance stage and is explicitly `not_run` in its record.
`render_checks.rgb_std` now measures the maximum **spatial** per-channel standard
deviation; colour differences within a single pixel no longer count as detail.
Alpha, coverage, variation and deformation thresholds are unchanged. Mesh drift
and deformation suitability (finding F6) remain open; no universal drift cutoff
or broadened character-quality claim is introduced by Batch 1.
`profile.json` selects backend settings; manifests pin source and model assets.
No personal inputs, generated characters, workstation paths or logs belong here.
