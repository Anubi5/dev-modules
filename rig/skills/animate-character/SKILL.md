---
name: animate-character
description: Create a bounded local walk preview from an EVA-validated named GLB rig.
---

Use `ai animate --help`; begin with `--duration 2` before a longer preview. The command requires the adjacent hash-bound `verify_glb.json` with passing finish and render checks. Revalidate old reports through the current rig verifier; never fabricate a success record or animate an unvalidated intermediate.

The existing Blender/ffmpeg route saves a scene, animated GLB, MP4, frames and provenance. Inspect frames at opposing stride phases and verify actual video FPS/frame count/duration. Check foot contact and deformations visually; deterministic keyframes are experimental motion, not production-quality acceptance. Only walk is implemented; Unreal handoff remains an alternative.
