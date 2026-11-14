"""Module manifest v2 contract (offline): schema, declared effects against the real commands, read-only proof.

A module opts in with "schema": 2 in module.json. Its effects must name exactly the commands in bin/ (or, for a
command with subcommands, "<command> <subcommand>" for each one its own help lists); a missing entry means the
command is not authorised. Subcommand names come from the module's optional tests/commands.py:

    def subcommands() -> dict[str, list[str]]                 # command -> subcommands read from its real parser/help
    def read_only_fixture(key: str, home: Path) -> list[str]  # argv proving a read_only entry in a disposable HOME

Every read_only entry runs once in a disposable HOME with the GPU lock held and systemctl recorded: it must not
write into HOME, start a service or wait for the GPU lock. The platform's Pydantic model (Scripts) is authoritative;
the published JSON Schema copy here catches structural errors before a release pin is selected.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path(__file__).resolve().with_name('module-manifest.schema.json')


def v2_modules():
    """Module folders whose module.json declares schema 2."""
    for path in sorted(ROOT.glob('*/module.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and data.get('schema') == 2:
            yield path.parent, data


def helper(module):
    """The module's tests/commands.py, or None."""
    path = module / 'tests' / 'commands.py'
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f'{module.name.replace("-", "_")}_commands', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def expected_keys(module):
    """Every command in bin/, split into its subcommands where the module's helper reads them from help."""
    found = helper(module)
    listed = found.subcommands() if found and hasattr(found, 'subcommands') else {}
    keys = set()
    for command in sorted(p.name for p in (module / 'bin').glob('*') if p.is_file()) if (module / 'bin').is_dir() else []:
        subs = listed.get(command)
        keys |= {f'{command} {s}' for s in subs} if subs else {command}
    return keys


class ManifestContract(unittest.TestCase):
    """One subtest per v2 module."""

    def test_schema(self):
        """Structural validation against the published schema (jsonschema is required in CI)."""
        try:
            import jsonschema
        except ImportError:
            if os.environ.get('REQUIRE_JSONSCHEMA'):
                raise
            self.skipTest('jsonschema not installed (CI installs it)')
        schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
        for module, data in v2_modules():
            with self.subTest(module=module.name):
                jsonschema.validate(data, schema)
                self.assertEqual(data['name'], module.name)

    def test_files_exist(self):
        """MCP commands, skills and the page service are this module's own files; every skill folder is declared."""
        for module, data in v2_modules():
            with self.subTest(module=module.name):
                for server in data.get('mcp', []):
                    self.assertTrue((module / server['command']).is_file(), server['command'])
                declared = set(data.get('skills', []))
                present = {f'skills/{p.parent.name}' for p in module.glob('skills/*/SKILL.md')}
                self.assertEqual(declared, present, 'skills in module.json must match skills/*/SKILL.md')
                service = (data.get('ui') or {}).get('service')
                if service:
                    self.assertTrue((module / f'{service}.service').is_file(), service)

    def test_effects_cover_exactly_the_commands(self):
        """No undeclared command (not authorised) and no entry for a command that does not exist."""
        for module, data in v2_modules():
            with self.subTest(module=module.name):
                declared = set(data.get('effects', {}))
                expected = expected_keys(module)
                self.assertEqual(sorted(expected - declared), [], 'commands without declared effects')
                self.assertEqual(sorted(declared - expected), [], 'effects for commands the module does not have')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'read-only proof needs Linux (flock, bash)')
    def test_read_only_commands_have_no_effects(self):
        """A read_only entry neither writes into HOME, starts a service nor takes the GPU lock."""
        import fcntl
        for module, data in v2_modules():
            keys = [k for k, effects in data.get('effects', {}).items() if effects == ['read_only']]
            if not keys:
                continue
            found = helper(module)
            self.assertTrue(found and hasattr(found, 'read_only_fixture'), f'{module.name}: read_only entries need tests/commands.py read_only_fixture')
            for key in keys:
                with self.subTest(module=module.name, command=key), tempfile.TemporaryDirectory() as temp, \
                        open('/tmp/ai-gpu.lock', 'a+') as lock:
                    temp = Path(temp)
                    home, fake = temp / 'home', temp / 'bin'
                    home.mkdir(); fake.mkdir()
                    calls = temp / 'calls'
                    (fake / 'systemctl').write_text(f'#!/bin/sh\necho "systemctl $*" >> {calls}\n')
                    (fake / 'systemctl').chmod(0o755)
                    argv = found.read_only_fixture(key, home)
                    before = snapshot(home)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        env = dict(os.environ, HOME=str(home), PATH=f'{fake}:{os.environ["PATH"]}', MOD=str(module), MOD_NAME=module.name)
                        result = subprocess.run(argv, cwd=home, env=env, capture_output=True, text=True, timeout=60)
                    except subprocess.TimeoutExpired:
                        self.fail(f'{key} waited (GPU lock or service)')
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(snapshot(home), before, f'{key} wrote into HOME')
                    starts = [c for c in (calls.read_text().splitlines() if calls.exists() else []) if ' start ' in f'{c} ' or ' restart ' in f'{c} ']
                    self.assertEqual(starts, [], f'{key} started a service')


def snapshot(home):
    """Every path under home with size and modification time."""
    return {str(p.relative_to(home)): (p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else None
            for p in sorted(home.rglob('*'))}


if __name__ == '__main__':
    unittest.main()
