"""!
@file mesh_check.py
@brief Checks a generated 3D model in Blender (headless): imports the GLB, measures the mesh and renders it from
       4 sides (front, right, back, left) with Cycles on the GPU, so the result can be looked at and image-checked.
@details Run as: blender -b --factory-startup --python-exit-code 1 --python mesh_check.py -- MODEL.glb OUT_PREFIX
         Writes OUT_PREFIX_front.png ... (transparent background, so img-check measures the object) and prints one line "MESH-CHECK {json}". Watertightness is reported, not
         required (open shapes such as cups or cloth are fine); a model with no faces fails.
"""
import json
import math
import sys
import time

import bmesh
import bpy
from mathutils import Vector

src, prefix = sys.argv[sys.argv.index("--") + 1:][:2]
t0 = time.time()
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
stats = {"objects": len(meshes), "vertices": 0, "triangles": 0, "non_manifold_edges": 0, "loose_parts": 0,
         "uv_layers": 0, "materials": [], "textures": []}
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for o in meshes:
    bm = bmesh.new(); bm.from_mesh(o.data); bm.transform(o.matrix_world)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)   # GLB splits vertices at texture seams: join them to measure
    stats["vertices"] += len(bm.verts)
    stats["triangles"] += sum(len(f.verts) - 2 for f in bm.faces)
    stats["non_manifold_edges"] += sum(1 for e in bm.edges if not e.is_manifold)
    seen, parts = set(), 0                       # connected pieces: many small ones = loose debris
    for v in bm.verts:
        if v.index in seen:
            continue
        parts += 1; stack = [v]; seen.add(v.index)
        while stack:
            u = stack.pop()
            for e in u.link_edges:
                w = e.other_vert(u)
                if w.index not in seen:
                    seen.add(w.index); stack.append(w)
    stats["loose_parts"] += parts
    for v in bm.verts:
        lo = Vector(map(min, lo, v.co)); hi = Vector(map(max, hi, v.co))
    stats["uv_layers"] = max(stats["uv_layers"], len(o.data.uv_layers))
    for m in o.data.materials:
        if m and m.use_nodes:
            stats["materials"].append(m.name)
            stats["textures"] += [f"{n.image.name} {n.image.size[0]}x{n.image.size[1]}" for n in m.node_tree.nodes
                                  if n.type == "TEX_IMAGE" and n.image]
    bm.free()
size = hi - lo
stats["size"] = [round(x, 3) for x in size] if meshes else None

prefs = bpy.context.preferences.addons["cycles"].preferences       # same device choice as the Blender module's test
used = "CPU"
for backend in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = backend; prefs.get_devices()
        if [d for d in prefs.devices if d.type == backend]:
            for d in prefs.devices:
                d.use = d.type == backend
            used = backend; break
    except Exception:
        pass
scene = bpy.context.scene
scene.render.engine = "CYCLES"; scene.cycles.device = "GPU" if used != "CPU" else "CPU"
scene.cycles.samples = 32; scene.render.film_transparent = True; scene.render.image_settings.color_mode = "RGBA"   # coverage = alpha
scene.render.resolution_x = scene.render.resolution_y = 768
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.58, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
renders = []
if meshes and stats["triangles"]:
    centre, radius = (lo + hi) / 2, max(size) * 0.5 or 1.0
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10)); bpy.context.object.data.energy = 3
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(60), 0, math.radians(45)))
    bpy.context.object.data.energy = 2
    bpy.ops.object.camera_add(); cam = bpy.context.object; scene.camera = cam
    cam.data.lens = 50
    dist = radius / math.tan(cam.data.angle / 2) * 1.25
    for name, ang in (("front", -90), ("right", 0), ("back", 90), ("left", 180)):
        a = math.radians(ang)
        cam.location = centre + Vector((math.cos(a) * dist, math.sin(a) * dist, radius * 0.35))
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{prefix}_{name}.png"
        bpy.ops.render.render(write_still=True)
        renders.append(scene.render.filepath)
stats.update(device=used, renders=renders, secs=round(time.time() - t0, 1), ok=bool(stats["triangles"]))
print("MESH-CHECK " + json.dumps(stats))
sys.exit(0 if stats["ok"] else 1)
