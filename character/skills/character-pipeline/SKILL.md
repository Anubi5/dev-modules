---
name: character-pipeline
description: Run and inspect EVA character pipeline stages using existing manifests and stage state.
---

Start with `ai character --help`, then inspect the existing project and manifests before generating. Use the pipeline's plan/run commands and existing stage state; preserve completed artifacts and input hashes. Use explicit model choices when identity or style matters and review any feature-default selection shown before generation.

The pipeline's animation stage is an Unreal handoff. For a local walk, use the rig module's `ai animate` only after finish and re-import validation pass. Do not mark an Unreal request complete because a local preview exists. Keep numerical validation and visual approval distinct; a failed image verifier is not a quality pass.
