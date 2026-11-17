"""Failed early validation preserves completed work and never starts skinning."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cli


class RecoveryTests(unittest.TestCase):
    def test_failed_orientation_retains_skeleton_and_skips_skin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / 'module'
            module.mkdir()
            for name in ('contract.json', 'profile.json', 'models.json', 'architecture.json'):
                (module/name).write_bytes((cli.MOD/name).read_bytes())
            profile = json.loads((module/'profile.json').read_text())
            models = json.loads((module/'models.json').read_text())
            (module/'app').mkdir()
            source = root/'source'
            (source/'configs/task').mkdir(parents=True)
            (source/'configs/model').mkdir()
            for task in profile['tasks'].values():
                (source/'configs/task'/task).write_text('{}')
            (source/'configs/model'/profile['model_config']).write_text('{}')
            (module/'app/source-path.txt').write_text(str(source))
            checkpoints = [dict(path=str(root/('checkpoint'+str(i)))) for i in range(len(models['weights']['files']))]
            (module/'app/installed.json').write_text(json.dumps(dict(checkpoints=checkpoints, architecture=dict(directory=str(root/'architecture')))))
            hashes = {item['path']: pin['sha256'] for item, pin in zip(checkpoints, models['weights']['files'])}
            hashes[str(root/'architecture/config.json')] = json.loads((module/'architecture.json').read_text())['sha256']
            input_file = root/'input.glb'
            input_file.write_bytes(b'input')
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                status = 0
                if any(str(x).endswith('prepare_mesh.py') for x in command):
                    output = Path(command[command.index('--')+2])
                    output.write_bytes(b'prepared')
                    Path(str(output)+'.json').write_text('{}')
                if any(str(x).endswith('run.py') for x in command):
                    output = next(x.split('=',1)[1] for x in command if x.startswith('--output='))
                    Path(output).write_bytes(b'completed skeleton')
                if any(str(x).endswith('check_orientation.py') for x in command):
                    status = 2
                return SimpleNamespace(pid=123, wait=lambda timeout: status)

            @contextlib.contextmanager
            def lease(*args):
                yield None
            fake_yaml = SimpleNamespace(safe_load=lambda value: {'llm': {}}, safe_dump=json.dumps)
            with patch.object(cli, 'MOD', module), patch.dict(os.environ, RIG_WORK_DIR=str(root/'work')), \
                    patch.object(cli, 'verify_source', return_value='source-hash'), \
                    patch.object(cli.importlib.metadata, 'version', return_value='fixture'), \
                    patch.object(cli, 'sha256', side_effect=lambda p: hashes.get(str(p), 'f'*64)), \
                    patch.object(cli.shutil, 'which', return_value='blender'), \
                    patch.object(cli, 'gpu_lease', lease), patch.object(cli, 'chat_state', return_value='inactive'), \
                    patch.object(cli, 'stop_chat'), patch.object(cli.VramMonitor, 'watch'), \
                    patch.object(cli.subprocess, 'Popen', side_effect=child), patch.dict('sys.modules', yaml=fake_yaml):
                args = cli.parser().parse_args([str(input_file), '--json'])
                self.assertEqual(cli.execute(args), 2, cli.RESULT)
            record = json.loads(next((root/'work/outputs/rig').glob('*/record.json')).read_text())
            self.assertEqual(record['status'], 'check_failed')
            self.assertEqual([s['name'] for s in record['stages']],
                             ['prepare', 'free_gpu', 'extract_mesh', 'skeleton', 'orientation'])
            self.assertEqual(record['outputs'], {})
            skeleton = record['intermediates']['skeleton.fbx']
            self.assertEqual(skeleton['status'], 'not_validated')
            self.assertEqual(Path(skeleton['path']).read_bytes(), b'completed skeleton')
            self.assertIn('not_validated', Path(skeleton['path']).parts)
            self.assertEqual(record['input_path'], str(input_file))
            self.assertIn('checkpoint_sha256', record)

    def test_merged_rig_is_retained_without_becoming_validated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'rigged.glb'
            source.write_bytes(b'completed merge')
            record = {'outputs': {}}
            cli.preserve_intermediate(record, root, source)
            self.assertEqual(record['outputs'], {})
            saved = record['intermediates']['rigged.glb']
            self.assertEqual(saved['status'], 'not_validated')
            self.assertEqual(Path(saved['path']).read_bytes(), source.read_bytes())
            self.assertEqual(saved['sha256'], cli.sha256(source))


if __name__ == '__main__':
    unittest.main()
