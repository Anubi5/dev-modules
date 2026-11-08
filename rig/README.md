# Rig

`ai rig model.glb --json` exposes a single `local-ai.producer/1` result on stdout;
progress goes to stderr. Human output remains the default. The character runner
requires this interface and creates an FBX-only rig hand-off; the producer record
retains the GLB and Blender outputs separately.

`bash rig/test.sh --full` runs offline checks first. Without `RIG_TEST_INPUT` it
prints an explicit GPU-acceptance skip and succeeds; a supplied missing/empty
fixture fails. Set that variable deliberately to run the real reference case.

**EXPERIMENTAL — manual use. F6: OPEN — DEFERRED.** Reference numerical
acceptance remains valid; visual quality and generalization to other characters
are not established. F6 is deliberately deferred and does not block experimental
development. Automatic crash recovery remains a limit for unattended operation.
Keep the recorded quality limits; do not treat a numerical pass as visual signoff.

Local humanoid rigging with [UniRig](https://github.com/VAST-AI-Research/UniRig)
and Blender, with an independent Python environment, pinned checkpoints,
shared GPU lease and a record for every started job. Repository updates do not
automatically install or replace the workstation's existing module.

**Lifecycle staging passed on one reference at commit `214dcf0`.** A real loaded
chat model was stopped during GPU rigging, a second rig job waited, normal exit
restored chat, and cancellation/external-watchdog interruption preserved the prior
service state. Parent SIGKILL left its GPU child holding the lease and its record
stale; supervised manual cleanup/restoration was required. This does not establish
automatic crash recovery.

**Functional asset staging passed for the retained reference; visual quality is
not accepted.** Real 90/180/270-degree rotations and a controlled asymmetric
variant passed finishing and FBX/GLB re-imports; wrong-facing and mirrored
transforms were rejected. These are transformations of one existing rig, not
independent characters or rotated-input inference tests. Updated Unreal walk/jog
exports passed all 102 frame checks at unchanged thresholds, with 1.700 m rest
height and twelve inspected renders. Maximum jog edge p99.9 was 2.472 against
the 2.5 limit. Visible shoulder/sleeve deformation remains a known quality
limitation (F6: OPEN — DEFERRED); numerical validation does not establish visual
acceptance. No executable module source changed during these sessions. Broader
fixture coverage and automatic recovery remain outstanding.

The F6 investigation traced the visible tears to joint smoothing on disconnected
UV/normal-split vertices: coincident copies start with equal weights and diverge
during smoothing, before Unreal export. Welding before smoothing preserves the
reference's triangles and face-corner UV data and closes the tears in controlled
retained-pose renders, but **is not an accepted fix**: walk/jog p99.9 stretch reaches
2.678/4.403 against the unchanged 2.5 limit. Post-smoothing weight matching and a
512-iteration welded profile also fail existing gates. These candidates were
rejected; executable module code is unchanged. F6 is OPEN — DEFERRED. Those failed
candidates are not part of the experimental module and must not replace it.

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
Automated stale-job recovery and native crash recovery remain outstanding.

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
