"""CPU-only early orientation check; no inference, rendering or asset edits."""
import json
from pathlib import Path
import sys
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orientation_blender import verify_armature


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    source = Path(args[0])
    facing = args[1] if len(args) > 1 else "-y"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    if source.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source))
    elif source.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source))
    else:
        raise ValueError("Orientation input must be FBX or GLB")
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    try:
        if len(arms) != 1:
            raise ValueError("Orientation needs exactly one armature")
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"
                  and any(m.type == "ARMATURE" and m.object == arms[0] for m in o.modifiers)]
        result = dict(ok=True, orientation=verify_armature(arms[0], meshes, facing))
    except ValueError as error:
        result = dict(ok=False, error=str(error))
    print("RIG-ORIENTATION " + json.dumps(result), flush=True)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
