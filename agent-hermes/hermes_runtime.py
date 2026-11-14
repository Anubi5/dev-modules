"""One installed-version check and official no-lazy-install policy for every EVA entry point.

Dependency preparation is an explicit owner install operation. Launch never repairs
an environment. This is configuration enforcement, not isolation from the same UID.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PINNED_REVISION='d923a5eb487a17c0a68801bba9ef0f56c81040a8'
LOCKS=('pyproject.toml','uv.lock','pm/pyproject.toml','pm/uv.lock','pm/lock.json')


def source_root():
    return Path.home()/'.hermes/hermes-agent'


def environment(env=None, *, install=False):
    result=dict(os.environ if env is None else env)
    result.update(HERMES_DISABLE_LAZY_INSTALLS='1',PYTHONDONTWRITEBYTECODE='1')
    if not install:
        result.update(UV_OFFLINE='1',PIP_NO_INDEX='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return result


def verify_installed(executable='hermes',source=None):
    """The only installed-Hermes version verification; reused at install/launch/setup/tests."""
    source=Path(source or source_root()).resolve()
    actual=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=PINNED_REVISION or dirty:raise ValueError('Hermes source differs from the reviewed installed version; review and run the install')
    binary=shutil.which(executable)
    if not binary:raise ValueError('Hermes launcher missing; run the install')
    path=Path(binary).resolve()
    if not path.is_relative_to(source):
        with path.open('rb') as stream:wrapper=stream.read(16384).decode('utf-8')
        if str(source/'.hermes/bin/hermes') not in wrapper:
            raise ValueError('Hermes launcher does not map to the reviewed installation')
    for name in LOCKS:
        if not (source/name).is_file():raise ValueError('Hermes locked dependency manifest missing; run the install')
    return dict(revision=actual,source=str(source),launcher=str(path))


def install_dependencies(executable='hermes',source=None):
    if os.environ.get('DEV_UNATTENDED'):raise ValueError('Hermes dependency installation requires an attended owner operation')
    verified=verify_installed(executable,source)
    source=Path(verified['source'])
    before={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in LOCKS}
    # Official explicit PM install uses the pinned app + PM locks. --extra all
    # prepares the application closure without installing optional default tools
    # or model assets. Do not run pm update, re-resolve pins, or install on launch.
    subprocess.run([executable,'pm','install','--extra','all'],env=environment(install=True),check=True)
    if before!={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in LOCKS}:
        raise ValueError('Hermes dependency locks changed during install; review required')
    return dict(status='prepared',**verified)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['check','install'])
    parser.add_argument('--hermes',default='hermes');parser.add_argument('--source',type=Path)
    args=parser.parse_args(argv)
    try:
        result=install_dependencies(args.hermes,args.source) if args.operation=='install' else verify_installed(args.hermes,args.source)
        print(json.dumps(result));return 0
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        print('CHECK Hermes: '+str(error)+'; dependencies must be prepared during the owner-started install');return 1


if __name__=='__main__':raise SystemExit(main())
