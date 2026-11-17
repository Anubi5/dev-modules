import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import animate


class AnimationTests(unittest.TestCase):
    def test_only_hash_bound_completed_validation_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'model.glb';source.write_bytes(b'validated fixture')
            report=source.with_name('verify_glb.json')
            good=dict(ok=True,errors=[],finish_passed=True,input_sha256=animate.sha256(source),render_checks=[dict(ok=True)])
            report.write_text(json.dumps(good));animate.validated_input(source,report)
            for change in (dict(ok=False),dict(errors=['bad weights']),dict(finish_passed=False),dict(render_checks=[]),dict(input_sha256='0'*64)):
                report.write_text(json.dumps(dict(good,**change)))
                with self.assertRaises(ValueError): animate.validated_input(source,report)
            report.write_text(json.dumps(good));source.write_bytes(b'changed after validation')
            with self.assertRaisesRegex(ValueError,'differs'): animate.validated_input(source,report)

    def test_limits_and_invalid_input_fail_before_service_or_output(self):
        for duration in (0,11,float('nan')):
            args=animate.parser().parse_args(['missing.glb']);args.duration=duration
            with patch.object(animate,'chat_state',side_effect=AssertionError('service reached')), self.assertRaises(ValueError):
                animate.execute(args)
        args=animate.parser().parse_args(['missing.glb'])
        with patch.object(animate.subprocess,'run',side_effect=AssertionError('process reached')), self.assertRaises(ValueError):
            animate.execute(args)

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'model.glb';source.write_bytes(b'rig')
            report=root/'verify_glb.json';report.write_text(json.dumps(dict(ok=True,errors=[],finish_passed=True,input_sha256=animate.sha256(source),render_checks=[dict(ok=True)])))
            out=root/'existing';out.mkdir();(out/'owner.txt').write_text('preserve')
            args=animate.parser().parse_args([str(source),'--output',str(out)])
            with patch.object(animate.shutil,'which',return_value='fixture'), \
                    patch.object(animate.subprocess,'run',return_value=SimpleNamespace(stdout=' V....D libx264 encoder')), \
                    patch.object(animate,'chat_state',side_effect=AssertionError('service reached')), self.assertRaises(FileExistsError):
                animate.execute(args)
            self.assertEqual((out/'owner.txt').read_text(),'preserve')


if __name__=='__main__': unittest.main()
