"""Deterministic walk using armature-space rotations converted to bone-local keys."""
import math
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from orientation import verify_facing, verify_named_sides


def main():
    source, directory, count = sys.argv[sys.argv.index('--')+1:]
    output=Path(directory);count=int(count)
    if not 24 <= count <= 240: raise ValueError('Walk preview must be 24-240 frames')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source)
    scene=bpy.context.scene
    arms=[o for o in scene.objects if o.type=='ARMATURE']
    if len(arms)!=1: raise ValueError('Expected one compatible armature')
    arm=arms[0]
    meshes=[o for o in scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==arm for m in o.modifiers)]
    if not meshes: raise ValueError('No skinned mesh')
    required=['root','pelvis','spine_01','head']+[stem+'_'+side for stem in ('upperarm','lowerarm','thigh','calf','foot','ball','hand') for side in ('l','r')]
    if any(name not in arm.data.bones for name in required): raise ValueError('Incompatible rig bone contract')
    positions={b.name:arm.matrix_world@b.head_local for b in arm.data.bones}
    verify_facing([positions['ball_'+s]-positions['foot_'+s] for s in ('l','r')],
                  [positions['thigh_'+s] for s in ('l','r')],
                  [o.matrix_world.to_3x3().determinant() for o in [arm]+meshes])
    verify_named_sides(positions)
    if arm.parent or min(arm.scale) <= 0 or max(arm.scale)-min(arm.scale) > 1e-5:
        raise ValueError('Expected an unparented rig with positive uniform object scale')
    for obj in list(scene.objects):
        if obj not in [arm]+meshes: bpy.data.objects.remove(obj,do_unlink=True)
    arm.animation_data_clear()
    for bone in arm.pose.bones:
        bone.matrix_basis=Matrix.Identity(4);bone.rotation_mode='QUATERNION'
    def key(name,rotation,frame):
        rest=arm.data.bones[name].matrix_local.to_3x3()
        pose=arm.pose.bones[name]
        pose.rotation_quaternion=(rest.inverted()@rotation@rest).to_quaternion()
        pose.keyframe_insert('rotation_quaternion',frame=frame)
    rx=lambda angle:Matrix.Rotation(math.radians(angle),3,'X')
    ry=lambda angle:Matrix.Rotation(math.radians(angle),3,'Y')
    rz=lambda angle:Matrix.Rotation(math.radians(angle),3,'Z')
    base=arm.location.copy()
    # Key every frame: deterministic sampling with no dependency on curve defaults.
    for frame in range(1,count+1):
        phase=2*math.pi*(frame-1)/24; swing=math.sin(phase)
        for side,sign in (('l',1),('r',-1)):
            key('thigh_'+side,rx(-20*sign*swing),frame)
            key('calf_'+side,rx(30*max(0,-sign*swing)),frame)
            key('upperarm_'+side,rx(14*sign*swing)@ry(22*sign),frame)
            key('lowerarm_'+side,rx(-12),frame)
        key('spine_01',rz(3*swing),frame)
        arm.location=base+Vector((0,-.35*(frame-1)/24,0))
        arm.keyframe_insert('location',frame=frame)
    arm.animation_data.action.name='walk'
    scene.frame_start,scene.frame_end,scene.render.fps=1,count,24
    scene.frame_set(1)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in [arm]+meshes: obj.select_set(True)
    bpy.context.view_layer.objects.active=arm
    bpy.ops.export_scene.gltf(filepath=str(output/'motion.glb'),use_selection=True,export_animations=True)
    bpy.ops.mesh.primitive_plane_add(size=30)
    ground=bpy.context.object
    material=bpy.data.materials.new('ground');material.diffuse_color=(.18,.22,.25,1);ground.data.materials.append(material)
    scene.world=bpy.data.worlds.new('world');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.55,.65,.8,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.8
    bpy.ops.object.light_add(type='SUN',rotation=(.5,-.4,-.5));bpy.context.object.data.energy=3
    bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera;camera.data.lens=48
    for frame in range(1,count+1):
        scene.frame_set(frame)
        target=arm.location+Vector((0,0,.9))
        camera.location=target+Vector((2.3,-4.6,1.1))
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.keyframe_insert('location',frame=frame);camera.keyframe_insert('rotation_euler',frame=frame)
    scene.render.engine='CYCLES';scene.cycles.samples=16;scene.cycles.use_denoising=True
    prefs=bpy.context.preferences.addons['cycles'].preferences
    device=None
    for backend in ('OPTIX','CUDA'):
        try:
            prefs.compute_device_type=backend;prefs.get_devices()
            if any(d.type==backend for d in prefs.devices):
                for d in prefs.devices: d.use=d.type==backend
                device=backend;break
        except Exception: pass
    if device is None: raise ValueError('No existing Cycles GPU device; bounded preview refused')
    scene.cycles.device='GPU'
    scene.render.resolution_x,scene.render.resolution_y=640,360
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.filepath=str(output/'frames/frame_')
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'scene.blend'))
    bpy.ops.render.render(animation=True)
    print('ANIMATION rendered',count,'frames on',device,flush=True)


if __name__=='__main__': main()
