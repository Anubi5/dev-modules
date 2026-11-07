"""3D inspection and validation verdicts; no execution or GPU ownership."""
import json
import os


QUALITY = {"normal": {"resolution": 1024, "faces": 300000, "texture": 2048},     # ~10 GB peak expected: headroom
           "high": {"resolution": 1536, "faces": 700000, "texture": 4096}}       # the template's own; 10.5 GB measured


def glb_stats(path):
    """!
    @brief Reads a GLB's header (no extra tools): mesh, triangles, vertices, materials, textures. Never raises: a
           missing, truncated or malformed file gives {"valid": False, "error": why}. @return dict
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(20)
            n = int.from_bytes(head[12:16], "little")
            if len(head) < 20 or head[:4] != b"glTF" or head[16:20] != b"JSON":
                return {"valid": False, "error": "not a GLB file"}
            if int.from_bytes(head[8:12], "little") > size or 20 + n > size:
                return {"valid": False, "error": "truncated (file shorter than its header says)"}
            j = json.loads(f.read(n))
        acc, tri, verts = j.get("accessors", []), 0, 0

        def count(i):                                  # an accessor's count; a bad index is an error, not acc[-1]
            if not isinstance(i, int) or not 0 <= i < len(acc) or not isinstance(acc[i].get("count"), int):
                raise IndexError(f"accessor {i} out of range")
            return acc[i]["count"]
        for m in j.get("meshes", []):
            for p in m.get("primitives", []):
                verts += count(p["attributes"]["POSITION"])
                tri += count(p["indices"]) // 3 if "indices" in p else 0
        mats = [sorted(k for k in ("baseColorTexture", "metallicRoughnessTexture") if k in m.get("pbrMetallicRoughness", {}))
                + (["normalTexture"] if "normalTexture" in m else []) + (["occlusionTexture"] if "occlusionTexture" in m else [])
                for m in j.get("materials", [])]
        return {"valid": True, "meshes": len(j.get("meshes", [])), "triangles": int(tri), "vertices": int(verts),
                "material_maps": mats, "textures": len(j.get("textures", [])), "size_mb": round(size / 1e6, 1)}
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        return {"valid": False, "error": f"malformed GLB: {type(e).__name__}: {str(e)[:150]}"}


EXIT = {"succeeded": 0, "failed": 1, "check_failed": 2, "check_skipped": 3}


def check_status(glb_info, blender_present, blender_rc=None, mesh=None, renders=0, imgcheck_rc=None):
    """!
    @brief The verdict on a generated 3D model from its required checks: a valid GLB with triangles; with Blender
           installed, a successful Blender run whose result says ok, 4 renders, and img-check passing all of them.
           Without Blender the model is usable but unchecked ("check_skipped"). @return status (see EXIT)
    """
    if not (glb_info.get("valid") and glb_info.get("triangles", 0) > 0):
        return "failed"
    if not blender_present:
        return "check_skipped"
    if blender_rc != 0 or not isinstance(mesh, dict) or not mesh.get("ok") or renders < 4 or imgcheck_rc != 0:
        return "check_failed"
    return "succeeded"
