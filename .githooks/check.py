"""Small offline checks against the index; shared verbatim by both EVA repositories."""
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

START = time.monotonic()


@contextmanager
def snapshot():
    """Allow Windows scanners a short grace period to release parsed files."""
    directory = tempfile.mkdtemp(prefix='eva-precommit-')
    try:
        yield directory
    finally:
        for attempt in range(4):
            try:
                shutil.rmtree(directory)
                break
            except OSError:
                if attempt == 3:
                    print(f'WARNING: temporary snapshot still locked: {directory}', file=sys.stderr)
                else:
                    time.sleep(0.1 * (attempt + 1))


def run(args, **kwargs):
    """Bound the complete subprocess budget; never invoke a shell implicitly."""
    return subprocess.run(args, check=True, timeout=max(0.1, 28 - (time.monotonic() - START)), **kwargs)


def optional(name):
    """Locate a native tool, excluding Windows' WSL launch aliases."""
    value = shutil.which(name)
    if value and 'windowsapps' not in value.lower() and not (
        os.name == 'nt' and name == 'bash' and 'git' not in value.lower()
    ):
        return value
    if os.name == 'nt' and name == 'bash':
        git = Path(shutil.which('git')).resolve()
        for candidate in (git.parent.parent / 'bin/bash.exe', git.parent.parent / 'usr/bin/bash.exe'):
            if candidate.is_file():
                return str(candidate)
    print(f'WARNING: {name} unavailable; its optional syntax check is skipped.', file=sys.stderr)
    return None


def main():
    """Check changed files and only the contracts their paths select."""
    run(['git', 'diff', '--cached', '--check'])
    changed = run(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'],
                  stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\0').split('\0')
    changed = [p for p in changed if p]
    # An index snapshot makes partially staged changes and staged deletions authoritative.
    with snapshot() as directory:
        root = Path(directory)
        run(['git', '-c', 'core.autocrlf=false', 'checkout-index', '--all', '--prefix=' + root.as_posix() + '/'])
        shells = []
        powershell = []
        for name in changed:
            path = root / name
            if path.is_symlink() or not path.is_file():
                continue
            data = path.read_bytes()
            if name.endswith('.py'):
                ast.parse(data, filename=name)
            elif name.endswith('.json'):
                json.loads(data)
            elif name.endswith(('.ps1', '.psm1', '.psd1')):
                powershell.append(str(path))
            elif name.endswith('.sh') or (data.startswith(b'#!') and b'bash' in data.split(b'\n', 1)[0]):
                shells.append(str(path))
        if shells:
            bash = optional('bash')
            if bash:
                for path in shells:
                    run([bash, '-n', path])
        if powershell:
            pwsh = optional('pwsh') or (optional('powershell') if os.name == 'nt' else None)
            if pwsh:
                script = "$bad=0; $input | ConvertFrom-Json | ForEach-Object { $e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($_,[ref]$null,[ref]$e); if($e){$e | Out-String | Write-Host; $bad=1} }; exit $bad"
                run([pwsh, '-NoProfile', '-NonInteractive', '-Command', script],
                    input=json.dumps(powershell), text=True)
        manifests = sorted({str(Path(p).parent) for p in changed if p.endswith('/module.json')})
        if manifests:
            if (root / 'tools/module_manifest.py').is_file():
                try:
                    import pydantic  # noqa: F401
                except ImportError:
                    print('WARNING: Pydantic unavailable; manifest schema check skipped.', file=sys.stderr)
                else:
                    run([sys.executable, '-B', str(root / 'tools/module_manifest.py'), *manifests], cwd=root)
            else:
                checks = ['ManifestContract.test_files_exist']
                try:
                    import jsonschema  # noqa: F401
                except ImportError:
                    print('WARNING: jsonschema unavailable; manifest schema check skipped.', file=sys.stderr)
                else:
                    checks.append('ManifestContract.test_schema')
                # Reuse only these two small existing metadata checks (seven manifests),
                # never their command execution/read-only integration tests.
                run([sys.executable, '-B', str(root / '_tests/manifest_contract_test.py'), *checks], cwd=root)
        if 'ai-sandbox/config/release.json' in changed:
            try:
                import pydantic  # noqa: F401
            except ImportError:
                print('WARNING: Pydantic unavailable; release schema check skipped.', file=sys.stderr)
            else:
                run([sys.executable, '-B', str(root / 'tools/release_contract.py')], cwd=root)
        # Existing narrow, pure offline contracts; never invoke shell-dependent test suites here.
        selected = []
        if any(p in changed for p in ('tools/module_manifest.py', 'tools/module-manifest.schema.json')):
            selected.append('tools/module_manifest.py')
        if 'comfyui/nodes.txt' in changed or 'comfyui/test_sam2_pin.py' in changed:
            selected.append('comfyui/test_sam2_pin.py')
        for test in selected:
            if test.startswith('tools/'):
                try:
                    import pydantic  # noqa: F401
                except ImportError:
                    print('WARNING: Pydantic unavailable; schema parity check skipped.', file=sys.stderr)
                    continue
            args = ['--check-schema'] if test.startswith('tools/') else []
            run([sys.executable, '-B', str(root / test), *args], cwd=root)
    print(f'PASS: staged local checks ({time.monotonic() - START:.2f}s)')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
