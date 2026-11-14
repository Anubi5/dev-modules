"""Hermes launch never invokes a dependency installer; explicit install uses reviewed locks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hermes_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_launch_environment_refuses_lazy_install_even_with_caller_override(self):
        env=runtime.environment({'HERMES_DISABLE_LAZY_INSTALLS':'0'})
        self.assertEqual(env['HERMES_DISABLE_LAZY_INSTALLS'],'1')
        self.assertEqual(env['UV_OFFLINE'],'1')

    def test_wrong_version_fails_before_launcher_or_dependency_execution(self):
        with patch.object(runtime.subprocess,'run',return_value=SimpleNamespace(stdout='wrong')) as calls, \
             patch.object(runtime.shutil,'which',side_effect=AssertionError('Unverified launcher')):
            with self.assertRaises(ValueError):runtime.verify_installed()
        self.assertTrue(all(call.args[0][0]=='git' for call in calls.call_args_list))

    def test_attended_install_is_pinned_explicit_and_does_not_update(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)
            for name in runtime.LOCKS:
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture pinned manifest')
            with patch.dict(os.environ,{},clear=True), \
                 patch.object(runtime,'verify_installed',return_value=dict(revision=runtime.PINNED_REVISION,source=str(source))), \
                 patch.object(runtime.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                runtime.install_dependencies()
            self.assertEqual(run.call_args.args[0],['hermes','pm','install','--extra','all'])
            self.assertEqual(run.call_args.kwargs['env']['HERMES_DISABLE_LAZY_INSTALLS'],'1')

    def test_unattended_install_stops_before_any_verification_or_effect(self):
        with patch.dict(os.environ,DEV_UNATTENDED='1'),patch.object(runtime,'verify_installed',side_effect=AssertionError('effect')):
            with self.assertRaises(ValueError):runtime.install_dependencies()

    def test_changed_lock_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)
            for name in runtime.LOCKS:
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
            def mutation(*args,**kwargs):(source/'uv.lock').write_text('changed')
            with patch.dict(os.environ,{},clear=True),patch.object(runtime,'verify_installed',return_value=dict(source=str(source))),patch.object(runtime.subprocess,'run',side_effect=mutation):
                with self.assertRaisesRegex(ValueError,'locks changed'):runtime.install_dependencies()


if __name__=='__main__':unittest.main()
