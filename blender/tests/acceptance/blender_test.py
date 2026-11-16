"""!
@file blender_test.py
@brief Headless Blender check: renders a small Cycles scene on the GPU (OptiX, then CUDA). Fails if only the CPU is available.

Run as: blender -b --factory-startup --python-exit-code 1 --python blender_test.py
Prints one line "BLENDER-TEST OK|FAILED device=... render=...s version=... file=..." and exits 0 on success.
"""
import bpy, time, os, sys

out = "/tmp/blender-test.png"
prefs = bpy.context.preferences.addons["cycles"].preferences
used = "CPU"
for backend in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            used = f"{backend} ({gpus[0].name})"
            break
    except Exception as e:
        print(f"BLENDER-TEST {backend} not available: {e}")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU" if used != "CPU" else "CPU"
scene.cycles.samples = 64
scene.render.resolution_x, scene.render.resolution_y = 960, 540
scene.render.filepath = out

# simple scene: ground, a few shiny objects, sun + sky
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
bpy.ops.mesh.primitive_plane_add(size=20)
for i, x in enumerate((-2.5, 0, 2.5)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(x, 0, 1))
    mat = bpy.data.materials.new(f"m{i}"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = ((1, .2, .1, 1), (.1, .6, 1, 1), (.9, .9, .2, 1))[i]
    bsdf.inputs["Metallic"].default_value = .6
    bpy.context.object.data.materials.append(mat)
bpy.ops.object.light_add(type="SUN", location=(4, -4, 8)); bpy.context.object.data.energy = 4
bpy.ops.object.camera_add(location=(0, -9, 3.5), rotation=(1.25, 0, 0)); scene.camera = bpy.context.object
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

t0 = time.time()
bpy.ops.render.render(write_still=True)
dt = time.time() - t0
ok = used != "CPU" and os.path.exists(out) and os.path.getsize(out) > 10000   # a CPU render does not count
print(f"BLENDER-TEST {'OK' if ok else 'FAILED'} device={used} render={dt:.1f}s version={bpy.app.version_string} file={out}")
sys.exit(0 if ok else 1)
