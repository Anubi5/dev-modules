"""Pinned install receipt and direct launch; no installer may run at startup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('mcp_runtime.py')
spec = importlib.util.spec_from_file_location('mcp_runtime', SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
REVISION = 'a' * 40
SOURCE = 'git+https://projects.blender.org/lab/blender_mcp.git@' + REVISION + '#subdirectory=mcp'


class PreparedRuntimeTests(unittest.TestCase):
    def fixture(self, root):
        module = root / 'module'; module.mkdir(); (module/'mcp-requirements.lock').write_text('# no transitive fixture packages\n')
        executable = root / 'prepared env/bin/blender-mcp'; executable.parent.mkdir(parents=True)
        executable.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(7)\n')
        executable.chmod(0o755)
        metadata = executable.parent.parent / 'lib/python3.12/site-packages/blender_mcp-1.0.3.dist-info'
        metadata.mkdir(parents=True)
        (metadata / 'direct_url.json').write_text(json.dumps(dict(url='https://projects.blender.org/lab/blender_mcp.git',vcs_info=dict(commit_id=REVISION))))
        (metadata / 'METADATA').write_text('Name: blender-mcp\nVersion: 1.0.3\n')
        return module, executable, metadata

    def test_prepare_preserves_plugin_fields_and_retains_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module, executable, metadata = self.fixture(root)
            plugin = root / 'plugin'; plugin.mkdir()
            data = dict(mcpServers=dict(blender=dict(command='uv',args=['tool','run','--from',SOURCE,'blender-mcp','--transport','stdio'],env={'FIXTURE':'preserved'}),other=dict(command='existing')))
            original = json.dumps(data).encode(); (plugin / 'mcp.json').write_bytes(original)
            result = runtime.prepare(module,executable,SOURCE,plugin)
            after = json.loads((plugin / 'mcp.json').read_text())
            self.assertFalse(result['runtime_installs'])
            self.assertEqual(after['mcpServers']['blender']['command'],'blender-mcp-prepared')
            self.assertEqual(after['mcpServers']['blender']['args'],['--transport','stdio'])
            self.assertEqual(after['mcpServers']['blender']['env'],{'FIXTURE':'preserved'})
            self.assertEqual(after['mcpServers']['other'],data['mcpServers']['other'])
            backups = list(plugin.glob('mcp.json.before-prepared-*'))
            self.assertEqual(len(backups),1); self.assertEqual(backups[0].read_bytes(),original)
            runtime.prepare(module,executable,SOURCE,plugin)
            self.assertEqual(len(list(plugin.glob('mcp.json.before-prepared-*'))),1)

    def test_legacy_absolute_wrapper_migrates_to_portable_bare_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);module,executable,_=self.fixture(root)
            plugin=root/'plugin';plugin.mkdir()
            data=dict(mcpServers=dict(blender=dict(type='stdio',command=str(module/'bin/blender-mcp-prepared'),args=['--transport','stdio'],env={'BLENDER_MCP_HOST':'127.0.0.1'})))
            original=json.dumps(data).encode();(plugin/'mcp.json').write_bytes(original)
            runtime.prepare(module,executable,SOURCE,plugin)
            server=json.loads((plugin/'mcp.json').read_text())['mcpServers']['blender']
            self.assertEqual(server['command'],'blender-mcp-prepared')
            self.assertFalse(any(character.isspace() or character in '/\\' for character in server['command']))
            self.assertEqual(server['args'],data['mcpServers']['blender']['args'])
            self.assertEqual(server['env'],data['mcpServers']['blender']['env'])
            self.assertEqual(next(plugin.glob('mcp.json.before-prepared-*')).read_bytes(),original)

    def test_wrong_pin_or_custom_command_is_preserved_before_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module, executable, metadata = self.fixture(root)
            with self.assertRaises(ValueError):runtime.prepare(module,executable,SOURCE.replace(REVISION,'b'*40))
            self.assertFalse((module / 'app').exists())
            plugin=root/'plugin';plugin.mkdir();original=b'{"blender":{"command":"owner-custom","args":[]}}'
            (plugin/'mcp.json').write_bytes(original)
            with self.assertRaises(ValueError):runtime.prepare(module,executable,SOURCE,plugin)
            self.assertEqual((plugin/'mcp.json').read_bytes(),original)
            self.assertFalse((module / 'app').exists())

    @unittest.skipUnless(os.name=='posix','POSIX direct executable transport')
    def test_direct_launch_preserves_arguments_exit_and_refuses_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            module, executable, metadata = self.fixture(Path(directory))
            runtime.prepare(module,executable,SOURCE)
            args=['','two words','Grüße','x;$(no-shell)']
            command=[sys.executable,'-B',str(SCRIPT),'run','--module',str(module),'--']
            p=subprocess.run(command+args,capture_output=True,text=True)
            self.assertEqual(p.returncode,7);self.assertEqual(json.loads(p.stdout),args)
            (metadata/'METADATA').write_text('changed package version')
            p=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(p.returncode,1);self.assertIn('dependency metadata changed',p.stderr)
            self.assertFalse(p.stdout)

    def test_missing_receipt_fails_without_creating_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            module=Path(directory)/'module';module.mkdir()
            p=subprocess.run([sys.executable,'-B',str(SCRIPT),'run','--module',str(module),'--','--help'],capture_output=True,text=True)
            self.assertEqual(p.returncode,1);self.assertFalse(p.stdout)
            self.assertFalse((module/'app').exists())

    def test_dependency_versions_must_match_declared_lock_before_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            module, executable, metadata = self.fixture(Path(directory))
            (module/'mcp-requirements.lock').write_text('missing-package==1.0\n')
            with self.assertRaisesRegex(ValueError,'declared lock'):runtime.prepare(module,executable,SOURCE)
            self.assertFalse((module/'app').exists())


if __name__=='__main__':unittest.main()
