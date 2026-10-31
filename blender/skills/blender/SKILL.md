---
name: blender
description: Create, edit or render 3D scenes and models headless with Blender Python scripts (blender -b).
---

# Blender (headless)
## Procedure
1. Write a Python script in the project (`~/work/projects/<name>/src/`); use `bpy`.
2. Run `blender -b [file.blend] --python script.py` (Cycles uses the GPU when `scene.cycles.device = 'GPU'`).
3. Save `.blend`, `.glb` or images into `~/work/projects/<name>/output/`; the user opens them with Windows Blender.
## Rules
- One heavy GPU job at a time: run `ai free` first if image or chat models are loaded.
