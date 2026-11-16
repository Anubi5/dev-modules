# Generation ownership and finite follow-up

V3.2.1 reviews only operations, families, features, generation, runtime, assets,
the module declaration and skills. The reviewed merged base is modules
`1fb50ec68239b0251521040e57fad9fbbb5e9c7e`; no incoming commits were present at start.

| Responsibility | Current owner | Target / minimum next change |
|---|---|---|
| CLI routing and common timeout/seed bounds | `gen.py`; shared timeout guard in `runtime.py`; rerun invokes the same guard | Keep common bounds before dispatch. Each operation declares its accepted options; no parser framework. |
| Image/edit, pose, video and 3D inputs | `operations/image.py`, `edit.py`, `pose.py`, `video.py`, `model3d.py`; replay in `rerun.py` | Keep operation-specific validation here. Image composition is intentionally shared with edit. |
| Architecture graphs | `families/{sdxl,zimage,flux3,fibo2,anima,moss,wan22}/workflow.py`; Krea's declarative package; 3D recipes associated by package data | Keep one workflow owner per architecture. Recipes remain reviewed source, never executable model metadata. |
| Variants | RealVis/Juggernaut/SDXL-base share the SDXL builder; Anima Turbo shares Anima; Pixal3D multiview is recipe data | Already data rather than duplicate implementations, but directories mix architecture and variant names. Consolidate data atomically, preserving logical model IDs. |
| Features | `features/`; family `CAPABILITIES`; generic composition in `generation/shared.py` | LoRA loader/output compatibility and fixed CFG are now family-declared. Make every remaining feature default explicit per family. |
| Files, notes, execution and evidence | `assets.py` owns headers/notes; `runtime.py` still combines lookup, LoRA policy, services/jobs and records | Remove the remaining architecture-specific model-folder inference using existing family part metadata; avoid a new plugin layer. |

Correctness fixes reject unsupported mode options before input/service access,
including explicit default-valued flags; timeout overrides use 60–7200 seconds
for all six routes. Reference editing rejects ignored `--size`; pose extraction
rejects ignored prompt/size/seed inputs. Fixed-CFG families reject `--cfg` and
unsupported negative prompts. Saved LoRA strengths use the CLI's finite -2…2
constraint. LoRA injection verifies family-declared loader classes and ports;
`--force-lora` cannot authorize an undeclared loader. Valid existing graphs are
preserved. No user notes, choices, records or weights are migrated or rewritten.

## Next batch, in priority order

1. **Explicit feature declarations.** Files: `generation/builders.py` and the six
   image-family `workflow.py` files. Declare negative/upscale/edit/expand and other
   supported options in each owner, then make generic defaults deny undeclared
   features. Acceptance: the current variant capability matrix stays identical,
   unsupported options fail before effects, and all 252 graph hashes remain equal.
2. **Variant directory consolidation.** Files: `generation/registry.py`,
   `generation/models.py`, `families/{realvis,juggernaut,sdxl-base,anima-turbo}/package.json`
   and their architecture directories. First consolidate just SDXL/Anima variant
   data with one atomic reader update. Preserve existing profile IDs, selected
   defaults, model pins and notes/acceptance keys; hash only the selected variant.
   Acceptance: same catalogue order, capabilities and graphs; changing an unrelated
   variant does not invalidate selected-variant evidence. No dual-layout adapter.
3. **Metadata-owned lookup.** Files: `runtime.py:model_path/resolve_loras`,
   `assets.py` and `features/lora.py`. Resolve model folders/loaders from declared
   parts, then move only pure LoRA selection policy to its feature owner with
   explicit inputs. Acceptance: saved notes/checkpoints/LoRA selection remain
   unchanged, bad strengths/loaders still fail, and filesystem/graph contracts pass.
4. **Independent installation decision.** Files: `module.json`, `skills/` and
   repository architecture documentation. Decide whether separate image/edit/video/3D
   install units have an actual consumer requirement. Current operations and skills
   are distinct but share one ComfyUI installation. Acceptance: a bounded decision
   with dependency/ownership contracts; no duplicated ComfyUI, generator or framework.

Deferred: GPU re-acceptance of V3.2.1, all-family/experimental coverage, output-quality
attestations, video prompt fidelity, black MCP screenshots, F6 Rig/Unreal quality,
hostile same-UID isolation and production cutover. The prior candidate demonstrated
selected image/edit/video/3D/Hermes workflows; that is not acceptance of every
model/feature or a deployment of these new source revisions.
