---
name: blender
description: Create, edit or render 3D scenes and models headless with Blender Python scripts (blender -b).
---

# Blender (headless)
## Procedure
1. Write a Python script in the project (`~/work/projects/<name>/src/`); use `bpy`.
2. Run `blender -b [file.blend] --python script.py` (Cycles uses the GPU when `scene.cycles.device = 'GPU'`).
3. Save `.blend`, `.glb` or images into `~/work/projects/<name>/output/`; the user opens them with Windows Blender.
4. **Check the render before you say it is done** - a file that exists can still be empty. Aim the camera with a
   constraint and light the scene, then measure the image in the same script:
   ```python
   cam = bpy.context.scene.camera                       # aim at the subject instead of guessing angles
   c = cam.constraints.new('TRACK_TO'); c.target = subject; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
   bpy.ops.object.light_add(type='SUN', location=(4, -4, 8)); bpy.context.object.data.energy = 3
   bpy.ops.render.render(write_still=True)
   import numpy as np
   img = bpy.data.images.load(bpy.context.scene.render.filepath); px = np.array(img.pixels[:]).reshape(-1, 4)
   print("RENDER_CHECK std", float(px[:, :3].std()))  # near 0 = flat, empty image: fix camera/light and render again
   ```
   Also check what was asked (e.g. a red object: enough pixels with red clearly above green and blue). Only report
   DONE when the check passes; otherwise say what is wrong.
## Rules
- One heavy GPU job at a time: run `ai free` first if image or chat models are loaded.
