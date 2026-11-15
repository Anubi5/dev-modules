"""Family ownership and generic dispatch regressions; no runtime, GPU, services or model files."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from generation import builders, registry, models, video


class FamilyIsolationTests(unittest.TestCase):
    def test_each_procedural_family_owns_its_code(self):
        for name, package in registry.packages().items():
            if package.get('builder'):
                with self.subTest(family=name):
                    implementation = registry.implementation(package['builder'])
                    path = Path(implementation.__file__)
                    self.assertEqual(path.parent.name, package['builder'])
                    self.assertEqual(path.name, 'workflow.py')
                    self.assertTrue(callable(implementation.build))

    def test_metadata_cannot_select_arbitrary_python(self):
        for name in ('../evil', '/tmp/evil.py', 'x.py', 'x/y', 'missing', 'C:\\evil'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                registry.implementation(name)

    def test_discovery_does_not_execute_external_family_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            family = root / 'krea2'
            family.mkdir()
            source = registry.ROOT / 'krea2/package.json'
            (family / 'package.json').write_bytes(source.read_bytes())
            (family / 'workflow.py').write_text("raise AssertionError('metadata executed code')")
            self.assertIn('krea2', registry.packages(root))

    def test_generic_operation_and_shared_code_have_no_family_literals(self):
        root = registry.ROOT.parent
        names = {'sdxl', 'zimage', 'flux3', 'fibo2', 'anima', 'moss', 'wan22'}
        for relative in ('operations/image.py', 'operations/video.py', 'generation/builders.py', 'generation/shared.py'):
            tree = ast.parse((root / relative).read_text())
            constants = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertFalse(constants & names, relative)

    def test_checkpoint_variants_share_one_architecture_implementation(self):
        selected = [registry.implementation(registry.family(models.profile(n))['builder'])
                    for n in ('realvis', 'juggernaut', 'sdxl-base')]
        self.assertIs(selected[0], selected[1])
        self.assertIs(selected[0], selected[2])

    def test_new_video_family_is_discovered_without_a_catalogue_entry(self):
        import subprocess
        import sys
        code = """from copy import deepcopy
from generation import registry
entries = registry.profiles()
new = deepcopy(entries['wan2.2-ti2v-5b'])
new['family'] = 'new-video'
entries['new-video'] = new
registry.profiles = lambda: entries
from generation import models
assert models.package('new-video', kind='video').settings['family'] == 'new-video'
"""
        subprocess.run([sys.executable, '-B', '-c', code], cwd=registry.ROOT.parent, check=True, timeout=10)

    def test_prompt_templates_belong_to_the_family_and_affect_acceptance_identity(self):
        import hashlib
        p=models.profile('flux3');capability=builders.capabilities(p)
        for path in capability['prompt_templates'].values():
            self.assertTrue(path.startswith('families/flux3/prompts/'))
            self.assertTrue(registry.reviewed_file(path).is_file())
        before=registry.implementation_identity(p)
        original=Path.read_bytes
        def changed(path):
            value=original(path)
            return value+b'changed reviewed instructions' if path.name=='t2i.txt' else value
        with patch.object(Path,'read_bytes',changed):
            self.assertNotEqual(registry.implementation_identity(p),before)
        for bad in ('../outside','/etc/passwd','C:\\outside'):
            with self.assertRaises(ValueError):registry.reviewed_file(bad)

    def test_video_operation_dispatches_through_family(self):
        selected = models.package(kind='video')
        implementation = registry.implementation(registry.family(selected.settings)['builder'])
        with patch.object(implementation, 'build', return_value={'family': 'called'}) as build:
            self.assertEqual(video.build('x', 'y', 1, 20, 33, 't'), {'family': 'called'})
            self.assertEqual(build.call_args.args[0].name, selected.name)


if __name__ == '__main__':
    unittest.main()
