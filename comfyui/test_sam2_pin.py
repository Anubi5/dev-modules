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
        self.install_loop(False)

    def test_existing_pin_unrelated_requirements_and_weights_are_preserved(self):
        self.install_loop(True)

    def install_loop(self, already_pinned):
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
                sam2 = 'git+https://github.com/facebookresearch/sam2' + ('@' + COMMIT if already_pinned else '')
                path.write_text(sam2 + '\nnumpy\n' if name == names[0] else 'numpy\n')
                originals[path] = path.read_bytes()
            (module / 'nodes.txt').write_text(''.join('https://fixture.invalid/' + name + ' ' + 'a' * 40 + '\n' for name in names))
            weight = root / 'protected.safetensors'
            weight.write_bytes(b'owner fixture')
            before = (weight.read_bytes(), weight.stat().st_mtime_ns)
            uv = bindir / 'uv'
            uv.write_text('#!' + sys.executable + '\nimport json,os,pathlib,sys\n'
                          'args=sys.argv[1:];req=pathlib.Path(args[args.index("-r")+1])\n'
                          'with open(os.environ["CALLS"],"a") as f: f.write(json.dumps(dict(args=args,requirements=req.read_text()))+"\\n")\n')
            uv.chmod(0o755)
            script = ('set -euo pipefail\nA=$MOD/app; cd "$A/ComfyUI"\npin=()\n'
                      'say() { :; }\nat_commit() { :; }\n' + loop)
            result = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                                    env=dict(os.environ, MOD=str(module), CALLS=str(root / 'calls'),
                                             PATH=str(bindir) + ':' + os.environ['PATH']))
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            for call, name in zip(calls, names):
                self.assertEqual(call['args'][call['args'].index('-r') + 1], 'custom_nodes/' + name + '/requirements.txt')
                self.assertEqual(call['requirements'], 'git+https://github.com/facebookresearch/sam2@' + COMMIT + '\nnumpy\n' if name == names[0] else 'numpy\n')
            for path, original in originals.items():
                expected = original if already_pinned else original.replace(b'git+https://github.com/facebookresearch/sam2\n', ('git+https://github.com/facebookresearch/sam2@' + COMMIT + '\n').encode())
                self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(before, (weight.read_bytes(), weight.stat().st_mtime_ns))


if __name__ == '__main__':
    unittest.main()
