"""Local memory setup uses only explicit apply and preserves existing fixture data."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hermes_local_setup as setup


class SetupTests(unittest.TestCase):
    def test_default_is_a_pure_plan(self):
        with patch.object(setup.subprocess,'run',side_effect=AssertionError('Plan invoked runtime')), \
             patch.object(setup,'private_memory',side_effect=AssertionError('Plan wrote files')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(setup.main([]),0)
        plan=json.loads(output.getvalue());self.assertFalse(plan['downloads'])
        self.assertFalse(plan['settings']['memory.user_profile_enabled'])

    def test_shared_version_failure_precedes_config_and_memory(self):
        with patch.object(setup.runtime,'verify_installed',side_effect=ValueError('wrong version')), \
             patch.object(setup,'private_memory',side_effect=AssertionError('Memory changed')):
            with self.assertRaises(ValueError):setup.apply('hermes',Path('unused'))

    def test_failed_config_write_restores_entire_private_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);config=home/'config.yaml';original=b'unknown_key: preserved\nmodel: {provider: original}\n'
            config.write_bytes(original);calls=[]
            def run(command,**kwargs):
                calls.append(command)
                self.assertEqual(command[:3],['hermes','config','set'])
                self.assertEqual(kwargs['env']['HERMES_DISABLE_LAZY_INSTALLS'],'1')
                config.write_text('partial: mutation')
                raise subprocess.CalledProcessError(1,command)
            with patch.dict(os.environ,HERMES_HOME=str(home)), patch.object(setup.subprocess,'run',side_effect=run), \
                 patch.object(setup,'private_memory'),patch.object(setup.runtime,'verify_installed',return_value={'revision':setup.runtime.PINNED_REVISION}):
                with self.assertRaises(subprocess.CalledProcessError):setup.apply('hermes',home)
            self.assertEqual(config.read_bytes(),original)
            backups=list(home.glob('config.yaml.before-eva-*'))
            self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),original)
            self.assertEqual(len(calls),1)
            if os.name=='posix':self.assertEqual(backups[0].stat().st_mode&0o777,0o600)

    def test_failed_readback_restores_backup_and_retains_it(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);config=home/'config.yaml';config.write_text('custom: preserved')
            def write(command,**kwargs):config.write_text('changed');return SimpleNamespace(returncode=0)
            with patch.dict(os.environ,HERMES_HOME=str(home)),patch.object(setup.subprocess,'run',side_effect=write), \
                 patch.object(setup,'private_memory'),patch.object(setup.runtime,'verify_installed',return_value={'revision':setup.runtime.PINNED_REVISION}), \
                 patch.object(setup.policy,'inspect',return_value={'errors':['fixture failure'],'warnings':[]}):
                with self.assertRaises(ValueError):setup.apply('hermes',home)
            self.assertEqual(config.read_text(),'custom: preserved')
            self.assertEqual(len(list(home.glob('config.yaml.before-eva-*'))),1)

    def test_apply_passes_strings_verbatim_and_keeps_typed_containers(self) -> None:
        """Match the pinned CLI's verbatim strings and structured-value coercion."""
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);config=home/'config.yaml';original=b'custom: preserved\n'
            config.write_bytes(original);stored={}
            def write(command,**kwargs):
                key,argument=command[3:]
                # Upstream DEFAULT_CONFIG string fields are preserved verbatim.
                expected=setup.settings()[key]
                stored[key]=argument if isinstance(expected,str) else json.loads(argument)
                return SimpleNamespace(returncode=0)
            with patch.dict(os.environ,HERMES_HOME=str(home)),patch.object(setup.subprocess,'run',side_effect=write), \
                 patch.object(setup,'private_memory'),patch.object(setup.runtime,'verify_installed',return_value={'revision':setup.runtime.PINNED_REVISION}), \
                 patch.object(setup.policy,'inspect',return_value={'errors':[],'warnings':[]}):
                self.assertEqual(setup.apply('hermes',home)['status'],'configured')
            self.assertEqual(stored,setup.settings())
            self.assertEqual(stored['model.provider'],'custom')
            self.assertEqual(stored['model.default'],'qwen3.5-9b-agent')
            self.assertEqual(stored['model.base_url'],'http://127.0.0.1:8080/v1')
            self.assertIs(stored['memory.user_profile_enabled'],False)
            self.assertIsInstance(stored['plugins.enabled'],list)
            self.assertEqual(config.read_bytes(),original)

    @unittest.skipUnless(os.name=='posix','POSIX permissions require Linux; no Windows policy changes')
    def test_memory_permissions_and_contents_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);root=home/'memories';root.mkdir()
            note=root/'MEMORY.md';note.write_text('owner-approved synthetic note')
            setup.private_memory(home)
            self.assertEqual(note.read_text(),'owner-approved synthetic note')
            self.assertEqual(root.stat().st_mode&0o777,0o700)
            self.assertEqual(note.stat().st_mode&0o777,0o600)


if __name__=='__main__':unittest.main()
