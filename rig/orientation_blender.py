"""Shared Blender-world orientation evidence for skeleton and finish checks."""
from orientation import verify_facing


def verify_armature(arm, meshes, facing="-y"):
    bones = arm.data.bones
    roots = [b for b in bones if b.parent is None]
    if len(roots) != 1 or len(roots[0].children) != 3:
        raise ValueError("Orientation needs one pelvis with a spine and two legs")
    heads = {b.name: arm.matrix_world @ b.head_local for b in bones}
    root = roots[0]
    spine = max(root.children, key=lambda b: heads[b.name].z)
    legs = [b for b in root.children if b != spine and heads[b.name].z < heads[root.name].z + .02]
    segments, hips = [], []
    for leg in legs:
        chain = [leg]
        while len(chain[-1].children) == 1:
            chain.append(chain[-1].children[0])
        if len(chain) != 4 or chain[-1].children:
            raise ValueError("Orientation needs two four-joint toe-ended leg chains")
        segments.append(heads[chain[3].name] - heads[chain[2].name])
        hips.append(heads[leg.name])
    result = verify_facing(segments, hips,
                           [o.matrix_world.to_3x3().determinant() for o in [arm] + meshes], facing)
    result["terminal_foot_segments"] = [list(v) for v in segments]
    return result
