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

    def test_video_operation_dispatches_through_family(self):
        selected = models.package(kind='video')
        implementation = registry.implementation(registry.family(selected.settings)['builder'])
        with patch.object(implementation, 'build', return_value={'family': 'called'}) as build:
            self.assertEqual(video.build('x', 'y', 1, 20, 33, 't'), {'family': 'called'})
            self.assertEqual(build.call_args.args[0].name, selected.name)


if __name__ == '__main__':
    unittest.main()
