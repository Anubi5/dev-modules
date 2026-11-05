"""Blender preprocessing: decimate the delivered mesh before rigging, preserving UVs.

Run with Blender --background --python-exit-code 1 --python prepare_mesh.py --
INPUT.glb OUTPUT.glb TRIANGLE_BUDGET. Writes OUTPUT.glb.json with geometry evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import bpy
from mathutils.bvhtree import BVHTree


def digest(file):
    h = hashlib.sha256()
    with open(file, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def surface(objects):
    vertices, faces = [], []
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        offset = len(vertices)
        vertices.extend(obj.matrix_world @ vertex.co for vertex in mesh.vertices)
        faces.extend(tuple(offset + i for i in tri.vertices) for tri in mesh.loop_triangles)
    return vertices, faces


def distances(points, other_vertices, other_faces):
    tree = BVHTree.FromPolygons(other_vertices, other_faces, all_triangles=True)
    sample = random.Random(12345).sample(points, min(5000, len(points)))
    return sorted(tree.find_nearest(point)[3] for point in sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("triangles", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.triangles < 1000 or args.output.exists():
        raise ValueError("Budget must be >=1000 and output must not already exist")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects or any(obj.type == "ARMATURE" for obj in bpy.context.scene.objects):
        raise ValueError("Expected an unrigged mesh: decimation must precede rigging")
    before, before_faces = surface(objects)
    if not before_faces:
        raise ValueError("Input has no triangles")
    height = max(v.z for v in before) - min(v.z for v in before)
    ratio = min(1.0, args.triangles / len(before_faces))
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if not obj.data.uv_layers or not obj.data.materials:
            raise ValueError("Textured mesh must retain UVs and materials")
        modifier = obj.modifiers.new("Triangle budget", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    after, after_faces = surface(objects)
    if len(after_faces) > args.triangles:
        raise ValueError(f"Decimation exceeded budget: {len(after_faces)} > {args.triangles}")
    drift = distances(before, after, after_faces) + distances(after, before, before_faces)
    drift.sort()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.output), use_selection=True, export_animations=False)
    report = {"input_sha256": digest(args.input), "output_sha256": digest(args.output),
              "blender": bpy.app.version_string, "budget": args.triangles,
              "triangles_before": len(before_faces), "triangles_after": len(after_faces),
              "vertices_before": len(before), "vertices_after": len(after),
              "sampled_surface_distance_p95_height_fraction": drift[int(.95 * (len(drift) - 1))] / max(height, 1e-12),
              "sampled_surface_distance_max_height_fraction": max(drift) / max(height, 1e-12),
              "uv_layers": {obj.name: len(obj.data.uv_layers) for obj in objects},
              "ok": True}
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MESH-BUDGET " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
