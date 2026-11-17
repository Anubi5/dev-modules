---
name: rig-character
description: Rig a fictional humanoid with EVA and inspect orientation, validation, and retained intermediates.
---

Use `ai rig --help` for current options. Reuse retained skeleton/GLB evidence before another inference run. Declare the actual facing; the supported toe-ended foot chains are checked before skinning. A wrong or ambiguous rig must fail rather than be accepted by changing its label.

Read the output `record.json`, finish checks and both re-import reports. Files under `not_validated/` preserve completed work and provenance but are not validated deliverables. A numerical pass does not establish shoulder deformation or visual quality. Inspect saved renders and keep original inputs. There is no automatic stage resume.
