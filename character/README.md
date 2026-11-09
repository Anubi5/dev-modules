# Character

Character pipeline: turns a description into an image, 3D model, rig and Unreal animation, step by step.

The runner requires `--json` from generation and rig producers. It reads one
`local-ai.producer/1` result, retains stderr as `producer.log` and stdout as
`producer.json`, and never discovers records by scraping progress text. Producers
that do not support this protocol fail explicitly. Prompts after options/`--`
are accepted by the generator's real argparse parser, including literal dashes.

New rig manifests are **FBX-only**, declaring cm/+Z/+X consumption space. GLB and
Blender outputs remain in the retained producer record; they are not assigned
the FBX coordinate declaration. Existing manifests remain readable for history.

Experimental, human-operated composition of the existing image/video, mesh and
rig commands. Python standard library only; no model installation, service,
framework or GPU ownership is introduced. Install this module through the normal
module manager when deliberately updating the workstation; repository merges do
not install it automatically.

```bash
ai character manifest image --file "front=$HOME/work/demo/front.png" --output ~/work/demo/front.asset.json
ai character inspect ~/work/demo/front.asset.json
ai character plan ~/work/demo/pipeline.json
ai character run ~/work/demo/pipeline.json --work-dir ~/work/demo
ai character run ~/work/demo/pipeline.json --work-dir ~/work/demo --resume
bash character/test.sh  # Offline contract and orchestration tests
```

Paths in `--file` must be under the output manifest's parent. Use explicit paths
or expand `$HOME`; a `~` after `=` is not expanded by every shell. Geometry
manifests also require `--units m|cm` and `--forward=-Y|+X`. Manifest creation
hashes files; it does not inspect geometry or grant numerical/visual acceptance.

## File contract: `local-ai.asset/1`

An artifact declares `kind` (`image`, `video`, `mesh`, `rig`, `animation`), uniquely
named file roles, portable relative paths, byte counts and SHA-256 hashes. Geometry
declares intended **consumption** coordinates: units `m`/`cm`, up `+Z`, forward
`-Y`/`+X`. These are the consuming tool's coordinate convention, not a claim about
raw FBX/glTF header axes. Importers remain responsible for format conversion.
The producer may include its source record and hash. Validation states are
separate: numerical `passed`/`failed`/`not_run`, visual `not_established`/`passed`/
`failed`, generalization `not_established`. Numerical producer claims do not
establish visual quality. Rig artifacts retain `experimental: true` and
**F6: OPEN — DEFERRED**. No rig-quality investigation is performed by this module.

Unknown schemas, changed/missing files, duplicate roles, traversal, absolute file
references and symlink escapes fail. Manifests are written atomically and never
silently replaced. Keep the manifest and its files together when transferring
between WSL and Windows; run `inspect` again after transfer.

## Pipeline contract: `local-ai.pipeline/1`

```json
{
  "schema": "local-ai.pipeline/1",
  "stages": [
    {"id":"mesh", "operation":"mesh", "inputs":[{"manifest":"front.asset.json"}],
     "parameters":{"model":"pixal3d"}},
    {"id":"rig", "operation":"rig", "inputs":[{"stage":"mesh"}],
     "parameters":{"height":1.7,"facing":"-y"}},
    {"id":"motion", "operation":"animation", "inputs":[{"stage":"rig"}]}
  ]
}
```

Image/video stages accept a prompt and the supported scalar generation parameters;
mesh stages accept image manifests, model/quality/seed/faces/texture settings; rig
stages accept one GLB, height/triangle/smoothing/facing/seed settings. Each producer
retains ownership of its GPU lease, dependencies and validation. The pipeline
owns its workspace lock, stage state, logs and copied artifacts. It invokes fixed
commands with argument arrays; metadata cannot choose a shell command or Python
entry point. `timeout_seconds` bounds each producer, up to 7200 seconds.

Successful artifacts are copied into retained stage directories and hashed.
`--resume` verifies the unchanged plan, inputs and completed artifact hashes, then
skips completed producers. Failed attempts remain retained. Cooperative
interruption stops the owned child process group and records interruption.
SIGKILL/host loss may leave a producer running: inspect/recover its process and
GPU lease manually before changing a retained `running` stage to `interrupted`.
The runner deliberately refuses automatic replay of such stages.

The Windows boundary is explicit: an animation stage writes
`STAGE.unreal-request.json` and returns `awaiting_host`. Run the Windows Unreal
command against that request, then resume. The response must be a versioned
animation manifest bound to the request SHA-256. The hardened WSL distro never
launches Windows programs. Human operators can also run stages separately and
create/inspect manifests; no unattended-recovery claim is made.
