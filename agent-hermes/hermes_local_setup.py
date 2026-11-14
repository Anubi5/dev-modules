"""Plan local Hermes memory/vision settings; apply only during an approved deployment.

No credentials, model acquisition, remote memory providers or dependency installers
are used. Existing memory contents are never read, replaced or deleted.
"""
import argparse
import json
import os
from pathlib import Path
import stat
import shutil
import tempfile
import subprocess
from typing import Any

import hermes_profile_check as policy
import hermes_runtime as runtime


def settings() -> dict[str,Any]:
    """Return the reviewed non-secret settings, including curated memory only."""
    return dict(policy.EXPECTED,**{'skills.disabled':sorted(policy.DISABLED_SKILLS)})


def private_memory(home: Path) -> Path:
    """Create/restrict an operator-owned memory directory without reading contents."""
    if os.name!='posix':raise RuntimeError('Memory permission acceptance requires Linux')
    home=Path(home)
    if home.is_symlink() or not home.is_dir() or home.stat().st_uid!=os.getuid():
        raise ValueError('Hermes home must be an existing operator-owned directory, not a symlink')
    root=home/'memories'
    if root.is_symlink():raise ValueError('Refusing symlink memory storage')
    if root.exists() and (not root.is_dir() or root.stat().st_uid!=os.getuid()):raise ValueError('Memory storage ownership/type mismatch')
    for name in ('MEMORY.md','USER.md'):
        path=root/name
        if path.is_symlink():raise ValueError('Refusing symlink memory file')
        if path.exists() and (not stat.S_ISREG(path.stat().st_mode) or path.stat().st_uid!=os.getuid()):
            raise ValueError('Memory file ownership/type mismatch')
    root.mkdir(mode=0o700,exist_ok=True);root.chmod(0o700)
    for name in ('MEMORY.md','USER.md'):
        path=root/name
        if path.exists():path.chmod(0o600)
    return root


def apply(executable: str,home: Path,source: Path|None=None,revision: str|None=None) -> dict[str,Any]:
    """Apply reviewed settings through the pinned official configuration CLI.

    Args:
        executable: Existing Hermes launcher.
        home: Explicitly selected private profile directory.
        source: Existing pinned Hermes source tree.
        revision: Optional assertion of the centrally reviewed revision.

    Returns:
        Configuration status, policy report and retained private backup path.

    Raises:
        ValueError: Installation, profile ownership or policy verification fails.
        subprocess.SubprocessError: An official configuration operation fails.
    """
    verified=runtime.verify_installed(executable,source)
    if revision is not None and revision!=runtime.PINNED_REVISION:
        raise ValueError('Use the centrally pinned Hermes revision')
    home=Path(home)
    if home.resolve()!=Path(os.environ.get('HERMES_HOME',Path.home()/'.hermes')).resolve():
        raise ValueError('Hermes home differs from selected profile; set HERMES_HOME explicitly')
    config=home/'config.yaml'
    if config.is_symlink() or not config.is_file():raise ValueError('Requires an existing regular config.yaml')
    if os.name=='posix' and (home.is_symlink() or home.stat().st_uid!=os.getuid() or config.stat().st_uid!=os.getuid()):
        raise ValueError('Hermes home/config ownership mismatch')
    mode=stat.S_IMODE(config.stat().st_mode)
    fd,name=tempfile.mkstemp(prefix='config.yaml.before-eva-',dir=home)
    backup=Path(name)
    with os.fdopen(fd,'wb') as target,config.open('rb') as original:
        shutil.copyfileobj(original,target)
    # The retained backup is private, including when the original was too open.
    backup.chmod(0o600)
    try:
        private_memory(home)
        for key,value in settings().items():
            # Hermes preserves string-typed CLI arguments verbatim; JSON quotes
            # would become part of the stored provider, model or endpoint.
            argument=value if isinstance(value,str) else json.dumps(value)
            subprocess.run([executable,'config','set',key,argument],env=runtime.environment(),capture_output=True,text=True,check=True,timeout=20)
        report=policy.inspect(executable,home/'SOUL.md')
        if report['errors'] or any(w!='Owned SOUL profile section changed or missing' for w in report['warnings']):
            raise ValueError('Applied settings failed read-back verification')
        return dict(status='configured',hermes_revision=verified['revision'],profile=report,backup=str(backup),
                    memory='builtin curated files; user profile collection disabled')
    except BaseException:
        fd,name=tempfile.mkstemp(prefix='config.yaml.restore-',dir=home)
        with os.fdopen(fd,'wb') as target,backup.open('rb') as original:
            shutil.copyfileobj(original,target)
        os.chmod(name,mode)
        os.replace(name,config)
        raise


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='mutating; requires separately approved installation')
    parser.add_argument('--hermes',default='hermes')
    parser.add_argument('--home',type=Path,default=Path.home()/'.hermes')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--reviewed-revision')
    args=parser.parse_args(argv)
    if not args.apply:
        print(json.dumps(dict(status='plan',settings=settings(),effects=['private local memory directory','selected Hermes configuration only'],downloads=False),indent=2));return 0
    try:print(json.dumps(apply(args.hermes,args.home,args.source,args.reviewed_revision)));return 0
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print(json.dumps(dict(status='failed',error=type(error).__name__,meaning='Inspect compatibility and retained private config backup; no model download was attempted')));return 1


if __name__=='__main__':raise SystemExit(main())
