"""Reviewed zimage graph construction and feature composition."""
from generation.models import PROFILES, profile
from features.face_fix import sdxl as face_fix
from features.pose import zimage as zimage_pose
from generation.shared import _loaders, _finish


def zimage(prompt, seed, steps, width, height, prefix, prof, upscale=False, hires=False, fix_faces=False, pose=None, pose_strength=1.0):
    """!
    @brief Z-Image Turbo text-to-image (8 steps, CFG 1, AuraFlow sampling shift 3 - official template).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "lumina2")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
               "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
               "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": prof.get("shift", 3.0)}}})
    model = ["7", 0]
    if pose:
        model = zimage_pose.control(wf, model, pose, prof["pose_patch"], pose_strength, prefix, ["3", 0], width, height)
    return _finish(wf, prof, seed, steps, model, ["4", 0], ["5", 0], ["6", 0], prefix, upscale, hires,
                   postprocess=(lambda graph: repair(graph, seed)) if fix_faces else None)

def repair(wf, seed):
    """Z-Image uses the existing SDXL crop repair path, only when explicitly requested."""
    wf['27'] = {'class_type': 'SaveImage', 'inputs': {'filename_prefix': wf['10']['inputs']['filename_prefix'] + '_base', 'images': ['9', 0]}}
    return face_fix.with_sdxl_refiner(wf, ['9', 0], wf['4']['inputs']['text'], seed,
                                     profile(PROFILES.get('default_sdxl', 'realvis')))


def report(prof, wf, options):
    """Family-specific execution evidence."""
    return {'face_refiner': {'model': PROFILES.get('default_sdxl', 'realvis'),
                            'model_file': wf['44']['inputs']['ckpt_name']} if options.fix_faces else None}


def build(p, v, o):
    return zimage(v['prompt'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p, upscale=o.upscale, hires=o.hires,
        fix_faces=o.fix_faces, pose="pending" if o.pose else None, pose_strength=v["pose_strength"])

CAPABILITIES = dict(build=build, negative=False, features=['hires','fix_faces','pose'], uploads={'pose':'30'}, report=report)
