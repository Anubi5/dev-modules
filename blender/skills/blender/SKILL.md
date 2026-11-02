---
name: blender
description: Create, edit or render 3D scenes and models headless with Blender Python scripts (blender -b).
---

# Blender (headless)
## Procedure
1. Write a Python script in the project (`~/work/projects/<name>/src/`); use `bpy`.
2. Run `blender -b [file.blend] --python script.py` (Cycles uses the GPU when `scene.cycles.device = 'GPU'`).
3. Save `.blend`, `.glb` or images into `~/work/projects/<name>/output/`; the user opens them with Windows Blender.
4. **Check the render before you say it is done** - a file that exists can still be empty:
   `img-check ~/work/projects/<name>/output/<file>.png` - quote its line in your answer. `BLANK` or colours that do
   not fit the request (e.g. no red for a red cube) = not done: fix and render again.
   The usual cause of an empty render is a camera looking the wrong way. Aim it with a constraint instead of
   computing angles (a camera looks along its -Z axis):
   ```python
   c = cam.constraints.new('TRACK_TO'); c.target = subject; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
   bpy.ops.object.light_add(type='SUN', location=(4, -4, 8)); bpy.context.object.data.energy = 3
   scene.render.filepath = '/home/dev/work/projects/<name>/output/<file>.png'
   bpy.ops.render.render(write_still=True)        # writes the file itself; no Render Result tricks
   ```
   (`img-check` comes with the comfyui module; without it, load the PNG in the same Blender script and print
   the standard deviation of `img.pixels` - near 0 means empty.)
## Rules
- One heavy GPU job at a time: run `ai free` first if image or chat models are loaded.
