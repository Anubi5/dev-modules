# Family implementations

Image and video packages share `generation.registry` discovery. Metadata stays inert:
`packages()` validates `*/package.json`; a declarative package runs its bounded recipe.
A procedural package names a builder identity, for example `realvis` names `sdxl`.
The loader resolves only `families/<identity>/workflow.py` inside the pinned source;
metadata cannot select a filesystem path or import code from model storage.

`operations.image.prepare()` validates the request and invokes `builders.build()`.
For RealVis, Juggernaut and SDXL Base, that calls `families/sdxl/workflow.py:build`,
which constructs the SDXL graph and explicitly composes the user's selected pose,
face, repair and refinement features. The reusable implementations stay in `features/`.
`generation/shared.py` supplies common sampling/output nodes without family switches.
Z-Image owns its SDXL face-refiner composition and its extra reporting. FLUX 3,
Fibo 2 and MOSS own their timeout defaults. Wan owns video construction in `wan22/`.
The existing 3D packages own their reviewed `recipes/*.json` association.

To add a declarative family, follow `krea2/package.json`: declare pinned components,
defaults, capabilities, a graph and bounded input bindings. No Python adapter or router
edit is needed. A new reviewed procedural architecture adds its own `workflow.py`
with `build` and `CAPABILITIES`; its package names that folder identity. Checkpoint
variants reference the same builder and change only components/defaults.

The compatibility imports in `generation/image.py` and `generation/video.py` retain
existing callers. They contain no graph construction. The legacy workflow baseline
contains 235 hashes; `runtime.all_workflows()` currently yields 252 variants including
family packages. Schema checks and graph hashes establish offline construction only;
real GPU output and attended visual acceptance remain separate release gates.
