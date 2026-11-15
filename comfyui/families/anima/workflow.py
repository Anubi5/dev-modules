"""Reviewed anima graph construction and feature composition."""
from features.pose import anima as anima_pose
from generation.shared import _loaders, _finish


def anima(prompt, negative, seed, steps, width, height, prefix, prof, pose=None, pose_strength=0.8, upscale=False):
    """!
    @brief Anima text-to-image (illustration/anime), optionally steered by a pose (DWPose skeleton -> LLLite pose patch).
    @return Workflow in ComfyUI API format.
    """
    wf = _loaders(prof, "stable_diffusion")
    wf.update({"4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
               "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
               "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}})
    model = ["1", 0]
    if pose:
        model = anima_pose.control(wf, model, pose, prof["pose_patch"], pose_strength, prefix)
    return _finish(wf, prof, seed, steps, model, ["4", 0], ["5", 0], ["6", 0], prefix, upscale)

def build(p, v, o):
    return anima(v['prompt'], v['negative'], v['seed'], v['steps'], v['width'], v['height'],
        v['prefix'], p, pose='pending' if o.pose else None, pose_strength=v['pose_strength'], upscale=o.upscale)

CAPABILITIES = dict(lora={'model': ['1', 0], 'clip': ['2', 0], 'loaders': {'1': ['UNETLoader'], '2': ['CLIPLoader']}}, build=build, features=['pose'], uploads={'pose':'30'})
