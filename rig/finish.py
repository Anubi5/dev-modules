"""! @brief One Blender run from an auto-rigged GLB to an Unreal-ready FBX: names the bones by the skeleton's structure
(Unreal mannequin names), checks the rig contract, scales to a real height, tests key poses and a walk cycle (edge
stretch and triangle collapse per frame), renders them, exports FBX (with the walk as an animation) and GLB.
Run: blender -b --factory-startup --python rig_finish.py -- RIGGED.glb OUT_PREFIX [HEIGHT_M]   Prints "RIG-FINISH {json}". """
import json, math, sys
from pathlib import Path
import bpy
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, prefix = args[:2]; height = float(args[2]) if len(args) > 2 else 1.70
smooth_iterations = int(args[args.index("--smooth-iterations") + 1]) if "--smooth-iterations" in args else 128
if not 1 <= smooth_iterations <= 1000:
    raise ValueError("Smoothing iterations must be between 1 and 1000")
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath=src)
sc = bpy.context.scene
arm = next(o for o in sc.objects if o.type == "ARMATURE")
# glTF imports create a bone-display Icosphere too; it must never set body height.
meshes = [o for o in sc.objects if o.type == "MESH" and
          any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
if not meshes:
    raise RuntimeError("No meshes skinned to the imported armature")
report = {"problems": [], "character_meshes": [o.name for o in meshes],
          "excluded_meshes": [o.name for o in sc.objects if o.type == "MESH" and o not in meshes]}
def bad(t): report["problems"].append(t)

# ---------- 1. names from structure ----------
B = arm.data.bones
W = arm.matrix_world
head = {b.name: W @ b.head_local for b in B}
kids = lambda b: list(b.children)
def chain(b):                       # follow single children until a branch or a leaf
    out = [b]
    while len(b.children) == 1:
        b = b.children[0]; out.append(b)
    return out, b
def desc(b): return 1 + sum(desc(c) for c in b.children)
root = next(b for b in B if b.parent is None)
up = Vector((0, 0, 1))
rc = kids(root)
spine0 = max(rc, key=lambda c: head[c.name].z)
legs = [c for c in rc if c is not spine0 and head[c.name].z < head[root.name].z + 0.02]
names = {root.name: "pelvis"}
# facing: toes point forward
toes = []
for l in legs:
    ch, _ = chain(l)
    if len(ch) >= 4:
        toes.append(head[ch[3].name] - head[ch[2].name])
toe = sum(toes, Vector()) if toes else Vector(); toe.z = 0
fwd = Vector((0, -1, 0))            # glTF front (+Z) is Blender -Y; Pixal3D models face it (renders confirm)
report["toe_direction"] = [round(x, 3) for x in (toe.normalized() if toe.length > 1e-6 else toe)]
right = fwd.cross(up).normalized()
centre = head[root.name]
side = lambda b: "r" if (head[b.name] - centre).dot(right) > 0 else "l"
# spine up to the branch that carries the arms
sp, branch = chain(spine0)
for i, b in enumerate(sp):
    names[b.name] = f"spine_{i + 1:02d}"
bc = kids(branch)
neck = min(bc, key=lambda c: abs((head[c.name] - head[branch.name]).dot(right))) if bc else None
arms = [c for c in bc if c is not neck]
if neck:
    nk, _ = chain(neck)
    for i, b in enumerate(nk):
        names[b.name] = "head" if i == len(nk) - 1 and len(nk) > 1 else f"neck_{i + 1:02d}"
for a in arms:
    s = side(a); ch, hand = chain(a)
    for b, n in zip(ch, ["clavicle", "upperarm", "lowerarm", "hand"]):
        names[b.name] = f"{n}_{s}"
    if len(ch) < 4:
        bad(f"arm {s}: {len(ch)} bones before the hand (needs clavicle, upper arm, lower arm, hand)")
    fingers = [chain(f)[0] for f in kids(ch[-1])]
    if fingers:
        thumb = min(fingers, key=lambda f: (head[f[0].name] - head[ch[-1].name]).length)
        rest = sorted([f for f in fingers if f is not thumb], key=lambda f: (head[f[0].name] - head[thumb[0].name]).length)
        for f, n in zip([thumb] + rest, ["thumb", "index", "middle", "ring", "pinky"]):
            for i, b in enumerate(f):
                names[b.name] = f"{n}_{i + 1:02d}_{s}"
for l in legs:
    s = side(l); ch, _ = chain(l)
    for b, n in zip(ch, ["thigh", "calf", "foot", "ball"]):
        names[b.name] = f"{n}_{s}"
for b in B:
    names.setdefault(b.name, "extra_" + b.name)
need = ["pelvis", "spine_01", "head", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
        "hand_l", "hand_r", "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r"]
got = list(names.values())
for n in need:
    if got.count(n) != 1:
        bad(f"contract: {n} found {got.count(n)} times")
if len(set(got)) != len(got):
    bad("contract: duplicate names")
old2new = dict(names)
for b in B:                                        # two steps so no new name meets an old one; vertex groups follow
    b.name = old2new[b.name] + "#"
for b in B:
    b.name = b.name[:-1]
groups = {g.name for o in meshes for g in o.vertex_groups}
missing = sorted(n for n in set(old2new.values()) if n not in groups)
if missing:
    report["bones_without_weights"] = missing[:20]
report["bones"] = {"total": len(B), "named": sum(1 for n in got if not n.startswith("extra_")),
                   "extra": sorted(n for n in got if n.startswith("extra_")), "fingers_per_hand":
                   {s: len({n.split('_')[0] for n in got if n.endswith('_' + s) and n.split('_')[0] in ('thumb','index','middle','ring','pinky')}) for s in 'lr'}}
report["facing"] = [round(x, 3) for x in fwd]

# ---------- 2. real height ----------
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c); lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
s = height / max(hi.z - lo.z, 1e-6)
tops = {o for o in [arm] + meshes}
tops = {o if o.parent is None else o.parent for o in tops}
while any(o.parent for o in tops):
    tops = {o.parent or o for o in tops}
for o in tops:                                      # scale about the origin, then put the feet on the ground
    o.matrix_world = Matrix.Translation((0, 0, -lo.z * s)) @ Matrix.Scale(s, 4) @ o.matrix_world
bpy.context.view_layer.update()
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c); lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
MID = (lo + hi) / 2
report["scale"] = round(s, 4); report["height_m"] = round(hi.z - lo.z, 3); report["feet_z"] = round(lo.z, 3)

if "--smooth-joints" in args[3:]:
    # Standard Blender weight operations, restricted to shoulders/hips in world space.
    centres = [arm.matrix_world @ arm.data.bones[n].head_local for n in
               ("upperarm_l", "upperarm_r", "pelvis", "thigh_l", "thigh_r")]
    radius = height * 0.14
    report["weight_smoothing"] = {"radius_m": radius, "factor": 0.5, "iterations": smooth_iterations, "meshes": []}
    for o in meshes:
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True); bpy.context.view_layer.objects.active = o
        for vertex in o.data.vertices:
            position = o.matrix_world @ vertex.co
            vertex.select = any((position - centre).length < radius for centre in centres)
        selected = [v.index for v in o.data.vertices if v.select]
        before_weights = [{g.group: g.weight for g in o.data.vertices[i].groups} for i in selected]
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.vertex_group_smooth(group_select_mode="ALL", factor=0.5, repeat=smooth_iterations, expand=0)
        bpy.ops.object.vertex_group_clean(group_select_mode="ALL", limit=1e-6, keep_single=True)
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        changed = 0
        for index, old in zip(selected, before_weights):
            new = {g.group: g.weight for g in o.data.vertices[index].groups}
            changed += any(abs(old.get(g, 0) - new.get(g, 0)) > 1e-6 for g in old.keys() | new.keys())
        max_influences = max(sum(g.weight > 1e-6 for g in v.groups) for v in o.data.vertices)
        if max_influences > 4 or not changed:
            raise RuntimeError(f"Weight smoothing failed: {max_influences} influences; {changed} changed vertices")
        report["weight_smoothing"]["meshes"].append({"name": o.name, "selected": len(selected), "changed": changed,
                                                     "max_influences": max_influences})
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True); bpy.context.view_layer.objects.active = arm

# ---------- 3. poses, walk, measures ----------
P = arm.pose.bones
def pb(n): return P.get(n)
def reset():
    for p in P:
        p.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
def turn(n, target, deg):          # turn bone n by deg degrees towards a world direction (about its own head)
    p = pb(n)
    if not p:
        return
    M = arm.matrix_world
    h = M @ p.head; d = (M @ p.tail - h).normalized()
    axis = d.cross(target.normalized())
    if axis.length < 1e-6:
        return
    R = Matrix.Rotation(math.radians(deg), 4, axis.normalized())
    Mi = M.inverted()
    p.matrix = Mi @ Matrix.Translation(h) @ R @ Matrix.Translation(-h) @ M @ p.matrix
    bpy.context.view_layer.update()
dg = lambda: bpy.context.evaluated_depsgraph_get()
def snapshot():
    out = []
    for o in meshes:
        evaluated = o.evaluated_get(dg()); me = evaluated.to_mesh(); m = evaluated.matrix_world
        try:
            out.append(([m @ v.co for v in me.vertices], [tuple(e.vertices) for e in me.edges],
                        [tuple(f.vertices) for f in me.polygons]))
        finally:
            evaluated.to_mesh_clear()
    return out
def area(co, f):
    a, b, c = co[f[0]], co[f[1]], co[f[2]]
    return ((b - a).cross(c - a)).length / 2
reset(); REST = snapshot()
DOM = []
for o in meshes:
    gn = {g.index: g.name for g in o.vertex_groups}
    DOM.append([gn.get(max(v.groups, key=lambda g: g.weight).group, "?") if len(v.groups) else "none" for v in o.data.vertices])
def measure(tag):
    cur = snapshot(); er = []; ar = []; where = {}
    for (c0, e0, f0), (c1, _, _), dom in zip(REST, cur, DOM):
        for a, b in e0:
            L0 = (c0[a] - c0[b]).length
            if L0 > 1e-7:
                r = (c1[a] - c1[b]).length / L0; er.append(r)
                if r > 1.5:
                    k = dom[a] if dom[a] == dom[b] else "/".join(sorted((dom[a], dom[b])))
                    where[k] = where.get(k, 0) + 1
        for f in f0[::3]:
            A0 = area(c0, f)
            if A0 > 1e-10:
                ar.append(area(c1, f) / A0)
    er.sort(); ar.sort()
    q = lambda v, p: round(v[min(len(v) - 1, int(p * len(v)))], 3)
    return {"pose": tag, "edge_p95": q(er, .95), "edge_p99": q(er, .99), "edge_max": round(er[-1], 2),
            "edges_over_1_5": round(sum(1 for r in er if r > 1.5) / len(er), 5), "area_p01": q(ar, .01),
            "stretched_at": dict(sorted(where.items(), key=lambda kv: -kv[1])[:6]),
            "pass": q(er, .95) < 1.5 and q(er, .999) < 2.5 and q(ar, .01) > 0.2}
F, Dn, Bk = fwd, -up, -fwd
POSES = {
    "arms_down": [("upperarm_l", Dn, 40), ("upperarm_r", Dn, 40)],
    "elbows_60": [("upperarm_l", Dn, 40), ("upperarm_r", Dn, 40), ("lowerarm_l", F, 60), ("lowerarm_r", F, 60)],
    "hip_30_knee_60": [("thigh_l", F, 30), ("calf_l", Bk, 60)],
    "spine_15": [("spine_02", F, 15)],
}
results = []
prefs = bpy.context.preferences.addons["cycles"].preferences; dev = "CPU"
for be in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = be; prefs.get_devices()
        if [d for d in prefs.devices if d.type == be]:
            for d in prefs.devices: d.use = d.type == be
            dev = be; break
    except Exception:
        pass
sc.render.engine = "CYCLES"; sc.cycles.device = "GPU" if dev != "CPU" else "CPU"; sc.cycles.samples = 16
sc.render.resolution_x, sc.render.resolution_y = 512, 640; sc.render.film_transparent = True
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.58, 1)
bpy.ops.object.light_add(type="SUN", location=(0, 0, 10)); bpy.context.object.data.energy = 3
bpy.ops.object.camera_add(); cam = bpy.context.object; sc.camera = cam; cam.data.lens = 50
renders = []
def shoot(tag, view="front"):
    mid = MID
    d = (F if view == "front" else right) * height * 2.6
    cam.location = mid + d + Vector((0, 0, height * 0.1))
    cam.rotation_euler = (mid - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{prefix}_{tag}_{view}.png"; bpy.ops.render.render(write_still=True); renders.append(sc.render.filepath)
shoot("rest"); shoot("rest", "side")
for tag, moves in POSES.items():
    reset()
    for n, t, a in moves:
        turn(n, t, a)
    results.append(measure(tag)); shoot(tag, "side" if "hip" in tag or "spine" in tag else "front")
# walk: 8 frames, legs and arms swing in opposite phase; keyed for the FBX
reset()
act_frames = 8
for p in P:
    p.rotation_mode = "QUATERNION"
walk = []
for f in range(act_frames):
    sc.frame_set(f + 1)
    ph = 2 * math.pi * f / act_frames; sw = math.sin(ph)
    reset()
    turn("upperarm_l", Dn, 40); turn("upperarm_r", Dn, 40)
    turn("thigh_l", F if sw > 0 else Bk, 25 * abs(sw)); turn("thigh_r", Bk if sw > 0 else F, 25 * abs(sw))
    turn("calf_l", Bk, 10 + 45 * max(0, -math.cos(ph))); turn("calf_r", Bk, 10 + 45 * max(0, math.cos(ph)))
    turn("upperarm_l", Bk if sw > 0 else F, 18 * abs(sw)); turn("upperarm_r", F if sw > 0 else Bk, 18 * abs(sw))
    turn("lowerarm_l", F, 25); turn("lowerarm_r", F, 25)
    m = measure(f"walk_{f}"); walk.append(m)
    for p in P:
        p.keyframe_insert("rotation_quaternion", frame=f + 1); p.keyframe_insert("location", frame=f + 1)
    if f in (0, 2):
        shoot(f"walk{f}", "side")
sc.frame_start, sc.frame_end = 1, act_frames
arm.animation_data.action.name = "walk_test" if arm.animation_data and arm.animation_data.action else None
worst = max(walk, key=lambda x: x["edges_over_1_5"])
results.append(dict(worst, pose=f"walk (worst: {worst['pose']})", pass_all=all(x["pass"] for x in walk)))
report["poses"] = results

# ---------- 4. export ----------
# Packed GLB images can have extensionless paths (Image_0), which Unreal's
# Interchange translator cannot identify even when their bytes are embedded.
texture_dir = Path(prefix).parent / "textures"
texture_dir.mkdir(exist_ok=True)
images = {node.image for o in meshes for mat in o.data.materials if mat and mat.use_nodes
          for node in mat.node_tree.nodes if node.type == "TEX_IMAGE" and node.image}
report["textures"] = []
for index, image in enumerate(sorted(images, key=lambda im: im.name)):
    target = texture_dir / f"texture_{index}.png"
    image.file_format = "PNG"
    image.save(filepath=str(target))
    image.filepath = str(target)
    report["textures"].append(str(target))
# The procedural clip was a test, not part of the character deliverable. Clearing
# it prevents evaluation during export from replacing the neutral bind pose.
arm.animation_data_clear()
sc.frame_set(1); reset()
report["procedural_action_excluded_from_export"] = True
# Export a centimetre scene with identity object transforms. Leaving metre-space
# object scales on the rig produced inconsistent Unreal animation-only bind data.
for o in meshes:
    if o.parent:
        world = o.matrix_world.copy(); o.parent = None; o.matrix_world = world
for o in [arm] + meshes:
    o.matrix_world = Matrix.Scale(100.0, 4) @ o.matrix_world
sc.unit_settings.system = "METRIC"
sc.unit_settings.scale_length = 0.01
bpy.ops.object.select_all(action="DESELECT")
for o in [arm] + meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
# Unreal's animation FBX root is imported as Blender's armature object. Keep a
# separate, unweighted root so the weighted pelvis survives as an actual bone.
bpy.ops.object.mode_set(mode="EDIT")
root_bone = arm.data.edit_bones.new("root")
root_bone.head = (0, 0, 0)
root_bone.tail = (0, 0, 10)
root_bone.use_deform = False
arm.data.edit_bones["pelvis"].parent = root_bone
bpy.ops.object.mode_set(mode="OBJECT")
report["export_root"] = "root"
report["bones"]["total"] = len(arm.data.bones)
report["bones"]["named"] += 1
report["export_units"] = "cm"
report["export_object_scales"] = {o.name: list(o.scale) for o in [arm] + meshes}
fbx = prefix + "_unreal.fbx"
bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
                         apply_scale_options="FBX_SCALE_ALL", bake_anim=False, bake_anim_use_all_actions=False,
                         bake_anim_use_nla_strips=False, path_mode="COPY", embed_textures=True, mesh_smooth_type="FACE")
bpy.ops.wm.save_as_mainfile(filepath=prefix + "_rig.blend")
for o in [arm] + meshes:
    o.matrix_world = Matrix.Scale(0.01, 4) @ o.matrix_world
sc.unit_settings.scale_length = 1.0
bpy.context.view_layer.update()
bpy.ops.export_scene.gltf(filepath=prefix + "_named.glb", use_selection=True, export_animations=False)
report["gltf_metres_scale"] = 0.01
report.update(fbx=fbx, renders=renders, device=dev,
              ok=not report["problems"] and all(p["pass"] for p in report["poses"]))
print("RIG-FINISH " + json.dumps(report))
sys.exit(0 if report["ok"] else 2)
