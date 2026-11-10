"""Exercise the real custom-node install loop without downloads or installers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parent
COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'


class Sam2PinTests(unittest.TestCase):
    def test_floating_requirement_is_pinned_before_uv_runs(self):
        self.install_loop('git+https://github.com/facebookresearch/sam2\nnumpy\n')

    def test_absent_changed_or_duplicate_requirement_is_refused_before_uv(self):
        for text in ['numpy\n', 'git+https://github.com/facebookresearch/sam2@' + COMMIT + '\n',
                     'git+https://github.com/facebookresearch/sam2@other\n',
                     'git+https://github.com/facebookresearch/sam2\n' * 2]:
            with self.subTest(requirement=text):
                self.install_loop(text, invalid=True)

    def test_failed_uv_preserves_upstream_and_removes_only_its_temporary_file(self):
        self.install_loop('git+https://github.com/facebookresearch/sam2\nnumpy\n', uv_fail=True)

    def install_loop(self, requirement, invalid=False, uv_fail=False):
        source = (MODULE / 'install.sh').read_text()
        loop = source.split('# custom nodes:', 1)[1].split('say "InsightFace', 1)[0]
        loop = '# custom nodes:' + loop
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / 'module'
            app = module / 'app'
            comfy = app / 'ComfyUI'
            bindir = root / 'bin'
            bindir.mkdir()
            module.mkdir()
            names = ['ComfyUI-Impact-Pack', 'OtherNode']
            originals = {}
            for name in names:
                node = comfy / 'custom_nodes' / name
                node.mkdir(parents=True)
                path = node / 'requirements.txt'
                path.write_text(requirement if name == names[0] else 'numpy\n')
                originals[path] = path.read_bytes()
            (module / 'nodes.txt').write_text(''.join('https://fixture.invalid/' + name + ' ' + 'a' * 40 + '\n' for name in names))
            weight = root / 'protected.safetensors'
            weight.write_bytes(b'owner fixture')
            before = (weight.read_bytes(), weight.stat().st_mtime_ns)
            uv = bindir / 'uv'
            uv.write_text('#!' + sys.executable + '\nimport json,os,pathlib,sys\n'
                          'args=sys.argv[1:];req=pathlib.Path(args[args.index("-r")+1])\n'
                          'with open(os.environ["CALLS"],"a") as f: f.write(json.dumps(dict(args=args,requirements=req.read_text()))+"\\n")\n'
                          'sys.exit(23 if os.environ.get("UV_FAIL")=="1" else 0)\n')
            uv.chmod(0o755)
            script = ('set -euo pipefail\nA=$MOD/app; C=$A/ComfyUI; cd "$C"\npin=()\n'
                      'say() { :; }\nat_commit() { :; }\n' + loop)
            result = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                                    env=dict(os.environ, MOD=str(module), CALLS=str(root / 'calls'),
                                             UV_FAIL='1' if uv_fail else '0',
                                             PATH=str(bindir) + ':' + os.environ['PATH']))
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)
            self.assertEqual(list(comfy.glob('custom_nodes/*/.eva-requirements.*')), [])
            self.assertEqual(before, (weight.read_bytes(), weight.stat().st_mtime_ns))
            if invalid:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('SAM2 requirement missing or changed', result.stderr)
                self.assertFalse((root / 'calls').exists())
                return
            self.assertEqual(result.returncode, 23 if uv_fail else 0, result.stderr)
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            self.assertEqual(len(calls), 1 if uv_fail else 2)
            for call, name in zip(calls, names):
                request_path = Path(call['args'][call['args'].index('-r') + 1])
                if name == names[0]:
                    self.assertEqual(request_path.parent, comfy / 'custom_nodes' / name)
                    self.assertTrue(request_path.name.startswith('.eva-requirements.'))
                else:
                    self.assertEqual(str(request_path), 'custom_nodes/' + name + '/requirements.txt')
                self.assertEqual(call['requirements'], 'git+https://github.com/facebookresearch/sam2@' + COMMIT + '\nnumpy\n' if name == names[0] else 'numpy\n')


if __name__ == '__main__':
    unittest.main()
