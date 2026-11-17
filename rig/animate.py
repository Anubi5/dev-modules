"""Bounded local walk preview from a hash-verified rig re-import report."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from cli import gpu_lease
from chat_service import chat_state, stop_chat, start_chat
from fetch_assets import sha256

MOD = Path(__file__).resolve().parent


def validated_input(source, report_path):
    if source.suffix.lower() != '.glb' or not source.is_file():
        raise ValueError('Expected a validated named rig .glb')
    report = json.loads(report_path.read_text())
    if report.get('ok') is not True or report.get('errors') != [] or report.get('finish_passed') is not True:
        raise ValueError('Rig must pass finish and independent re-import validation')
    if report.get('input_sha256') != sha256(source):
        raise ValueError('Rig differs from the independently validated artifact; rerun verify.py')
    renders = report.get('render_checks')
    if not isinstance(renders, list) or not renders or any(row.get('ok') is not True for row in renders):
        raise ValueError('Rig requires passing saved render checks')
    return report


def parser():
    p = argparse.ArgumentParser(description='Render a deterministic local walk from a validated named rig; experimental motion quality')
    p.add_argument('input', type=Path)
    p.add_argument('--motion', choices=['walk'], default='walk')
    p.add_argument('--duration', type=float, default=10, help='seconds, 1-10; 24 fps')
    p.add_argument('--validation', type=Path, help='defaults to verify_glb.json beside the rig')
    p.add_argument('--output', type=Path, help='new output directory; existing paths are refused')
    p.add_argument('--timeout', type=int, default=600, help='Blender time limit in seconds, 60-900')
    return p


def execute(args):
    if not 1 <= args.duration <= 10 or not 60 <= args.timeout <= 900:
        raise ValueError('Duration must be 1-10 seconds and timeout 60-900 seconds')
    source = args.input.expanduser().resolve()
    validation = (args.validation or source.with_name('verify_glb.json')).expanduser().resolve()
    validated_input(source, validation)
    blender, ffmpeg = shutil.which('blender'), shutil.which('ffmpeg')
    if not blender or not ffmpeg:
        raise ValueError('Existing Blender and ffmpeg are required; nothing will be installed')
    encoders = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True, check=True, timeout=10)
    if 'libx264 ' not in encoders.stdout:
        raise ValueError('The existing ffmpeg must provide libx264')
    work = Path(os.environ.get('RIG_WORK_DIR', os.environ.get('WORK_DIR', str(Path.home()/'work')))).expanduser()
    job = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output = (args.output or work/'outputs/animation'/job).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'frames').mkdir()
    record = dict(schema='local-ai.animation/1', status='running', input=str(source), input_sha256=sha256(source),
                  validation=str(validation), validation_sha256=sha256(validation), motion=args.motion,
                  fps=24, frames=round(args.duration*24), outputs={}, visual_review='not_run',
                  code_sha256={name:sha256(MOD/name) for name in ('animate.py','animate_scene.py')})
    record['duration_seconds'] = record['frames']/24
    def save():
        pending=output/'record.json.new';pending.write_text(json.dumps(record,indent=2));pending.replace(output/'record.json')
    save()
    try:
        with gpu_lease() as fd:
            active = chat_state() == 'active'
            record['chat_was_active'] = active; save()
            try:
                stop_chat()
                subprocess.run(['ai','free'],check=True,timeout=45)
                env=dict(os.environ,AI_GPU_LOCK_FD=str(fd))
                with (output/'blender.log').open('w') as log:
                    subprocess.run([blender,'-b','--factory-startup','--disable-autoexec','--python-exit-code','1',
                                    '--python',str(MOD/'animate_scene.py'),'--',str(source),str(output),str(record['frames'])],
                                   env=env,pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
            finally:
                if active:
                    record['chat_restored']=False; save()
                    start_chat();record['chat_restored']=True; save()
        with (output/'encode.log').open('w') as log:
            subprocess.run([ffmpeg,'-nostdin','-v','error','-framerate','24','-i',str(output/'frames/frame_%04d.png'),
                            '-frames:v',str(record['frames']),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',
                            str(output/'preview.mp4')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        for name in ('scene.blend','motion.glb','preview.mp4'):
            file=output/name
            if not file.is_file() or not file.stat().st_size: raise ValueError('Missing animation artifact: '+name)
            record['outputs'][name]=dict(sha256=sha256(file),bytes=file.stat().st_size)
        record['status']='succeeded'
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='failed',error=str(error))
    finally:
        save()
    print(json.dumps(dict(status=record['status'],record=str(output/'record.json'),output=str(output),error=record.get('error'))))
    return 0 if record['status']=='succeeded' else 1


def main():
    try: return execute(parser().parse_args())
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        print('Animation error: '+str(error),file=sys.stderr);return 1


if __name__ == '__main__':
    sys.exit(main())
