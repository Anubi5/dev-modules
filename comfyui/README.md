# ComfyUI generation

Local image, editing, pose, video and textured 3D workflows using installed model files. Requires the declared ComfyUI/node/dependency pins, suitable CUDA resources and an EVA-compatible runner. See the [module contract](../README.md) for installation and model approval.

## Use and inspect

```bash
ai gen models
ai gen image 'a ceramic teapot in soft window light' --model zimage --seed 42 --name teapot
ai gen image 'a ceramic teapot' --model zimage --count 3 --seed 42 --name study
ai gen edit 'make the jacket red' --image "$HOME/work/inputs/person.png" --model flux3
ai gen pose --from "$HOME/work/inputs/pose.jpg" --name standing
ai gen video 'waves rolling onto a beach' --frames 33 --name waves
ai gen 3d --model pixal3d --image "$HOME/work/inputs/object.png" --quality normal
ai gen jobs 10
```

Use `ai gen <command> --help` for parser options and `ai gen show/check` for model metadata/readiness. Inputs must be inside `~/work`; copy Windows files there first. Profiles/packages declare capabilities: a parsed option is not supported by every model. Default image is Z-Image; SDXL-only controls without an explicit model select the configured SDXL fallback. `--count` is image-only, 1?8 sequential runs with incrementing seeds; failures retain completed outputs.

Explicit Z-Image supports native Union pose control, `--fix-faces` through the configured installed SDXL refiner, and `--hires`. Face identity (`--face`) and hand repair remain SDXL features. `--pose-skeleton` means the input is already a skeleton; otherwise extraction is required. Guides are resized without cropping, so use a compatible canvas ratio. Missing components fail before generation rather than downloading.

`--upscale` saves a fast 2? copy; `--hires` refines a larger copy on supported architectures. Preserve the base and inspect repairs for identity/anatomy drift. Editing supports FLUX 3/Fibo and up to four ordered references. Compatible LoRAs may be selected by name; discovery does not install architecture support.

## One character in several poses

This recipe is also consumed by the agent evaluation workflow. Select one clear fictional adult face reference, repeat the same hair/outfit description, and make full-body pose sources with head/feet visible. Use SDXL face/pose control and inspect every guide/result:

```bash
ai gen image 'full-body adult explorer, olive jacket, blue jeans, standing' --model realvis --face "$HOME/work/inputs/face.png" --pose "$HOME/work/inputs/pose.png" --fix-faces --fix-hands --hires --seed 300 --name explorer
```

Clothing consistency comes primarily from the repeated prompt; face guidance is not a universal identity lock. For overlapping limbs or difficult seated poses, compare FLUX 3 editing with the character and pose as two references. Review full-body framing before 3D reconstruction.

## Results and resource ownership

Results, records and exact API workflows reside under `~/work/outputs/comfy/<date>/`. `ai gen rerun <workflow.json>` verifies recorded inputs/model content; `--allow-drift` deliberately relaxes that check and `--fast` uses cached metadata. Neither proves unchanged visual quality. `--json` emits one `local-ai.producer/1` result on stdout; progress goes to stderr.

GPU leases, cancellation, retries, jobs and records belong to `gen.py`. Run one heavy job at a time; browser queues and fresh chat requests can compete. On host loss, inspect live processes/leases before replaying a retained running job. Do not delete locks blindly. 3D returns 0 for checked success, 1 for failure, 2 for failed validation and 3 for produced-but-unchecked output; review multi-view renders and geometry, not just file existence.

`pixal3d-mv` takes views in front/left/back/right order. Normal/high quality and explicit face/texture controls trade resources for detail; more polygons do not repair inconsistent views. Image/mesh output is not a production-ready rig. `img-check` supplies pixel checks; `img-ask` from llama-swap supplies a fallible semantic check. Human visual review remains necessary.

## Models, dependencies and offline operation

`models.txt`, `download_licenses.json` and family manifests declare model acquisition; `helper_models.json` pins DWPose/InsightFace helpers. The legacy helper list remains pin provenance, not an acquisition path. Model storage is `$MODELS_MOUNT/comfy`; approved writes use `$MODELS_RW`. Existing changed files fail without replacement. Automatic-download approval covers the declared personal, non-commercial policy, not broader rights. Restricted/gated files remain manual; read the recorded upstream terms before use.

`nodes.txt` pins executable custom nodes and `COMFY_REF` in `install.sh` pins ComfyUI. Python constraints are retained in the installed app. Review upgrades deliberately; ComfyUI-Manager is not installed. Pickle-based weights can execute code; hashes establish selected content, not safety of untrusted content.

ComfyUI disables pinned-memory reading by default because host-mounted weight reads can fail with that path. `~/.config/comfyui/override.env` can set `COMFY_MEM` for a controlled experiment; `memtest.py` compares behavior. Runtime/discovery never installs models. Application data in `~/work/comfyui` and outputs survive module removal.

## Extend and test

`generation/graph.py` binds recipes/LoRAs, `models.py` owns profiles/packages, and image/video/mesh components construct graphs or inspect data without executing jobs. Preserve `tests/workflow-baseline.json`; change graph baselines only for intentional reviewed behavior changes. Complex architectures use narrow builders; compatible variants should reuse profiles/recipes.

A reviewed image family lives in `families/<name>/package.json` (`local-ai.family/1`) with parts, sizes/hashes, defaults, capabilities, license/source references and a bound API graph. Validate bindings against `schema.json` and test real required files/workflow. Family metadata is trusted source, never an arbitrary Python entry point. Removing a package leaves weights untouched. Krea's package is an example; its manifest points to the governing upstream terms.

`ai models list/show/verify` is the shared catalogue. Discovery does not hash whole files, download, load models or infer visual quality. `verify` explicitly checks pins or fingerprints unpinned files; those outcomes differ. Unrecognized files may need support and are never presumed unused.

```bash
python3 -B comfyui/check_workflows.py
python3 -B -m unittest discover -s comfyui -p 'test_generation.py'
python3 -B -m unittest discover -s comfyui -p 'test_zimage.py'
```

Run module `test.sh --full` only with its real workstation prerequisites. Offline graph/contract checks do not replace GPU/visual acceptance. See [root guidance](../README.md) for parser-export and model-plan checks.
