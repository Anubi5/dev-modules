"""!
@file video.py
@brief ai gen video: a short clip from a prompt with the configured video package.
"""
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import runtime as rt
from generation.models import PROFILES, package
from generation import registry, builders
from generation.video import build


OPTIONS = {'frames', 'model', 'steps', 'negative', 'image', 'mp4'}


def validate(a):
    """! @brief Video's own limit, checked with the general ones before anything else. """
    if getattr(a, 'mp4', False):
        executable = shutil.which('ffmpeg')
        if not executable:
            sys.exit('--mp4 requires an existing ffmpeg; nothing will be installed')
        result = subprocess.run([executable, '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=10)
        if result.returncode or 'libx264 ' not in result.stdout:
            sys.exit('--mp4 requires libx264 in the existing ffmpeg')
    if len(a.image) > 1:
        sys.exit("video accepts at most one --image")
    if not 5 <= a.frames <= rt.MAX_FRAMES:
        sys.exit(f"--frames must be 5-{rt.MAX_FRAMES}")


def prepare(a, seed, prefix):
    """! @return (workflow, uploads, info, time limit, None) """
    frames = max(5, (a.frames - 1) // 4 * 4 + 1)
    if a.lora or a.pose_skeleton or a.pose or a.face or a.fix_faces or a.fix_hands or a.hires or a.checkpoint:
        sys.exit("--pose/--face/--fix-faces/--fix-hands/--hires/--checkpoint are for images")
    selected = package(a.model, kind='video')
    prof = dict(selected.settings, name=selected.name)
    if not getattr(a, 'check', False):
        registry.check_parts(prof, rt.MODELS)
    if a.image and not builders.capabilities(prof).get('image_conditioning'):
        sys.exit("Selected video family does not support --image")
    reference = rt.check_input(a.image[0]) if a.image else None
    uploads = [("11", reference)] if reference else []
    neg = a.negative if a.negative is not None else "static, blurry, distorted, low quality, watermark"
    wf, limit = build(a.prompt, neg, seed, a.steps or 20, frames, prefix, model=a.model, image="pending" if reference else None), a.timeout or 3600
    info = {"kind": "video", "prompt": a.prompt, "negative": neg, "seed": seed, "model": a.model or PROFILES.get("default_video", "wan2.2-ti2v-5b"),
            "steps": a.steps or 20, "frames": frames, "fps": prof["fps"],
            "duration_seconds": frames/prof["fps"], "image": reference}
    return wf, uploads, info, limit, prof


def export_mp4(files, fps):
    """Decode animated WebP with existing Pillow; ffmpeg only encodes raw frames."""
    from PIL import Image, ImageSequence
    converted = list(files)
    for name in files:
        source = Path(name)
        if source.suffix.lower() != '.webp': continue
        target = source.with_suffix('.mp4')
        if target.exists(): raise ValueError('Refusing to overwrite '+str(target))
        with Image.open(source) as image, tempfile.TemporaryFile(dir=source.parent) as raw:
            width,height=image.size
            count=0
            for frame in ImageSequence.Iterator(image):
                count+=1
                if count > rt.MAX_FRAMES or frame.size != (width,height):
                    raise ValueError('Unexpected video frame count or dimensions')
                raw.write(frame.convert('RGB').tobytes())
            if count < 2: raise ValueError('Expected an animated WebP')
            raw.seek(0)
            try:
                subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pixel_format','rgb24',
                                '-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0',
                                '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],
                               stdin=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=60)
            except subprocess.SubprocessError as error:
                raise RuntimeError('ffmpeg encoding failed; original WebP retained') from error
        converted.append(str(target))
    if len(converted)==len(files): raise ValueError('No animated WebP output to convert')
    return converted
