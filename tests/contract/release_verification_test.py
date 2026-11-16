"""Release runner outcomes using disposable test files, never installed runtimes."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class ReleaseRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = load('eva_runner', ROOT/'tests/run.py')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'tests/unit').mkdir(parents=True)
        self.report = self.root/'report.json'

    def entry(self, source, cases=None, platform='any', requires=None):
        path = 'tests/unit/test_fixture.py'
        (self.root/path).write_text(source, encoding='utf-8')
        row = dict(id=path, path=path, owner='core', category='unit', entrypoint=True,
                   platform=platform, requires=requires or [], case_names=cases or [])
        (self.root/'tests/inventory.json').write_text(json.dumps(dict(tests=[row])))

    def run_fixture(self, *args):
        with patch.object(self.runner, 'ROOT', self.root), patch.object(sys, 'argv', ['run.py','--report',str(self.report),*args]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.runner.main()

    def test_strict_passes_complete_execution(self):
        self.entry('import unittest\nclass T(unittest.TestCase):\n def test_ok(self): pass\n', ['T.test_ok'])
        self.assertEqual(self.run_fixture('--strict'), 0)
        self.assertEqual(json.loads(self.report.read_text())['results'][0]['passed_cases'], 1)

    def test_strict_fails_partial_but_ordinary_run_remains_permissive(self):
        self.entry('import unittest\nclass T(unittest.TestCase):\n @unittest.skip("fixture")\n def test_skip(self): pass\n', ['T.test_skip'])
        self.assertEqual(self.run_fixture(), 0)
        self.assertEqual(self.run_fixture('--strict'), 1)
        self.assertEqual(json.loads(self.report.read_text())['statuses']['PARTIAL'], 1)

    def test_strict_distinguishes_platform_skip(self):
        self.entry('', platform='linux' if sys.platform=='win32' else 'windows')
        self.assertEqual(self.run_fixture('--strict'), 1)
        self.assertEqual(json.loads(self.report.read_text())['results'][0]['skip_kind'], 'platform')

    def test_strict_fails_missing_dependency(self):
        self.entry('', requires=['eva_nonexistent_test_dependency'])
        self.assertEqual(self.run_fixture('--strict'), 1)
        self.assertEqual(json.loads(self.report.read_text())['results'][0]['skip_kind'], 'dependency')

    def test_child_stub_is_not_successful_release_evidence(self):
        self.entry('print("STUB fixture")')
        self.assertEqual(self.run_fixture('--strict'), 1)
        self.assertEqual(json.loads(self.report.read_text())['statuses']['SKIP'], 1)

    def test_zero_selection_reports_nonzero(self):
        self.entry('')
        self.assertEqual(self.run_fixture('--match','absent'), 2)
        self.assertEqual(json.loads(self.report.read_text())['selected'], 0)

    def test_missing_and_nested_unregistered_tests_are_rejected(self):
        self.entry('')
        (self.root/'tests/unit/test_fixture.py').unlink()
        with self.assertRaises(SystemExit): self.run_fixture('--strict')
        self.entry('')
        extra = self.root/'tests/acceptance/nested/test_unregistered.py'
        extra.parent.mkdir(parents=True); extra.write_text('')
        with self.assertRaises(SystemExit): self.run_fixture('--strict')

    def test_collection_cannot_pass_release_verification(self):
        self.entry('import unittest\nclass T(unittest.TestCase):\n def test_ok(self): pass\n', ['T.test_ok'])
        self.assertEqual(self.run_fixture('--strict','--collect'), 1)

class ReleaseHookTests(unittest.TestCase):
    def setUp(self):
        self.hook = load('eva_hook', ROOT/'.githooks/check.py')

    def test_optional_omissions_are_explicit_and_strict_omissions_fail(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.hook.skipped('fixture validator absent')
            self.assertEqual(self.hook.SKIPPED, ['fixture validator absent'])
            self.hook.RELEASE = True
            with self.assertRaisesRegex(ValueError, 'required release validator'):
                self.hook.skipped('fixture validator absent')

    def test_native_powershell_fallback_is_allowed(self):
        # A missing pwsh must not reject a usable native Windows PowerShell.
        with patch.object(self.hook.shutil, 'which', return_value=None):
            self.assertIsNone(self.hook.optional('pwsh'))
            self.assertEqual(self.hook.SKIPPED, [])


    def test_missing_schema_validator_cannot_report_release_pass(self):
        import builtins
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'ai-sandbox/config/release.json';path.parent.mkdir(parents=True);path.write_text('{}')
            def command(argv, **kwargs):
                return SimpleNamespace(stdout=b'ai-sandbox/config/release.json\0')
            original=builtins.__import__
            def importing(name,*args,**kwargs):
                if name=='pydantic': raise ImportError('fixture unavailable')
                return original(name,*args,**kwargs)
            with patch.object(self.hook,'run',side_effect=command), patch.object(self.hook,'snapshot',return_value=contextlib.nullcontext(directory)), patch.object(builtins,'__import__',side_effect=importing), contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                with patch.object(sys,'argv',['check.py']): self.hook.main()
                self.assertIn('INCOMPLETE:',output.getvalue())
                self.assertNotIn('PASS:',output.getvalue())
                with patch.object(sys,'argv',['check.py','--release']):
                    with self.assertRaisesRegex(ValueError,'required release validator'): self.hook.main()

if __name__ == '__main__':
    unittest.main()
