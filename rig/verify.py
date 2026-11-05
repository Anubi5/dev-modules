"""Blender-side independent re-import validation of an exported FBX or GLB."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_contract import load_contract, validate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=float, default=1.7)
    parser.add_argument("--renders", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.input.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.input), automatic_bone_orientation=False)
    else:
        bpy.ops.import_scene.gltf(filepath=str(args.input))
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise ValueError("Expected exactly one armature")
    arm = arms[0]
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and
              any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
    if not meshes:
        raise ValueError("No skinned character mesh")
    minimum, maximum, triangles, unweighted, influences, sum_error = [], [], 0, 0, 0, 0
    root_weighted = False
    for obj in meshes:
        obj.data.calc_loop_triangles()
        triangles += len(obj.data.loop_triangles)
        xyz = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", xyz)
        matrix = np.array(obj.matrix_world)
        z = xyz.reshape((-1, 3)) @ matrix[2, :3] + matrix[2, 3]
        minimum.append(float(z.min())); maximum.append(float(z.max()))
        names = {g.index: g.name for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            weights = [g for g in vertex.groups if g.weight > 1e-8]
            influences = max(influences, len(weights))
            unweighted += not weights
            sum_error = max(sum_error, abs(sum(g.weight for g in weights) - 1))
            root_weighted |= any(names[g.group] == "root" for g in weights)
    unit = bpy.context.scene.unit_settings.scale_length
    bones = arm.data.bones
    summary = dict(bones=[b.name for b in bones],
                   root_parent=bones["root"].parent.name if bones.get("root") and bones["root"].parent else None,
                   pelvis_parent=bones["pelvis"].parent.name if bones.get("pelvis") and bones["pelvis"].parent else None,
                   root_deforms=bool(root_weighted), max_influences=influences,
                   unweighted_vertices=int(unweighted), max_weight_sum_error=sum_error,
                   triangles=triangles, height_m=(max(maximum)-min(minimum))*unit,
                   requested_height_m=args.height, uv_layers=min(len(o.data.uv_layers) for o in meshes),
                   materials=min(len(o.data.materials) for o in meshes))
    summary["errors"] = validate_summary(summary, load_contract())
    if args.renders:
        report = json.loads(args.renders.read_text())
        summary["render_checks"] = []
        for file in report["renders"]:
            image = bpy.data.images.load(file, check_existing=False)
            rgba = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(rgba); rgba = rgba.reshape((-1,4))
            visible = rgba[:,3] > .05
            coverage = float(np.mean(visible))
            rgb_std = float(np.std(rgba[visible,:3])) if np.any(visible) else 0
            ok = coverage > .02 and rgb_std > .005
            summary["render_checks"].append(dict(file=file, coverage=coverage, rgb_std=rgb_std, ok=ok))
            if not ok: summary["errors"].append("Blank/empty render: " + Path(file).name)
            bpy.data.images.remove(image)
    summary["ok"] = not summary["errors"]
    args.output.write_text(json.dumps(summary, indent=2))
    print("RIG-VERIFY " + json.dumps(summary), flush=True)
    if not summary["ok"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
