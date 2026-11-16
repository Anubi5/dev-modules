"""Fail before input access, outputs, model inspection or GPU/service ownership."""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import assets
import gen
import runtime
from generation import models
from operations import rerun


class OptionLimitsTests(unittest.TestCase):
    def reject(self, arguments, message):
        with patch.object(sys, 'argv', ['gen.py', *arguments]), \
             patch.object(runtime, 'generate', side_effect=AssertionError('GPU reached')), \
             patch.object(gen.model3d, 'run', side_effect=AssertionError('3D dispatch reached')), \
             patch.object(runtime, 'resolve_model', side_effect=AssertionError('model inspected')), \
             contextlib.redirect_stderr(io.StringIO()) as output, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                gen.main()
        self.assertIn(message, str(error.exception) + output.getvalue())

    def test_all_modes_reject_out_of_bounds_timeout_before_dispatch(self):
        for mode in gen.GENERATION_KINDS:
            for timeout in ('0', '59', '7201', '-1'):
                with self.subTest(mode=mode, timeout=timeout):
                    self.reject([mode, '--timeout', timeout], '--timeout')

    def test_rerun_checks_timeout_before_opening_saved_workflow(self):
        for timeout in ('0', '59', '7201', '-1'):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(SystemExit, '--timeout'):
                rerun.rerun(['missing.workflow.json', '--timeout', timeout])
        self.assertEqual(runtime.validate_timeout(60), 60)
        self.assertEqual(runtime.validate_timeout(7200), 7200)

    def test_mode_incompatible_flags_even_when_equal_to_defaults(self):
        cases = {'image': [('--frames','33'), ('--frames','0'), ('--quality','normal'), ('--image','x'), ('--from','x')],
                 'edit': [('--frames','33'), ('--style','art'), ('--size','1024x1024'), ('--count','1'), ('--faces','100')],
                 'pose': [('--steps','8'), ('--cfg','1'), ('--upscale',), ('--negative','x'), ('--model','zimage'), ('--from','x','--seed','7'), ('--from','x','--size','1024x1024')],
                 'video': [('--size','1024x1024'), ('--upscale',), ('--cfg','1'), ('--quality','normal'), ('--improve-prompt',), ('--face-weight','0.65')],
                 '3d': [('--steps','8'), ('--frames','33'), ('--size','1024x1024'), ('--lora','x'), ('--cfg','1')]}
        for mode, options in cases.items():
            for option in options:
                with self.subTest(mode=mode, option=option):
                    self.reject([mode, *option], option[0])

    def test_fixed_cfg_is_family_declared_including_krea(self):
        for name in ('krea2','zimage','flux3','fibo2'):
            with self.subTest(model=name), patch.object(runtime, 'find_loras', return_value=[]):
                with self.assertRaisesRegex(SystemExit, '--cfg is not supported'):
                    runtime.resolve_loras(SimpleNamespace(prompt='x',lora=[],cfg=2,negative=None),models.profile(name),'')

    def test_notes_and_cli_use_the_same_strength_constraint(self):
        for value in (float('nan'),float('inf'),-2.01,2.01,True,None,'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                assets.lora_strength(value)
        for value in (-2,0,0.8,2):
            self.assertEqual(assets.lora_strength(value),float(value))
        for value in (float('nan'),float('inf'),-3,3,True,'invalid'):
            with self.subTest(note=value), patch.object(runtime,'find_loras',return_value=[('x.safetensors',None)]), \
                 patch.object(runtime,'LORA_DIRS',['/fixture']), patch.object(runtime.os.path,'exists',return_value=True), \
                 patch.object(runtime,'model_path',return_value=None), patch.object(assets,'note_for',return_value={'strength':value}), \
                 self.assertRaisesRegex(SystemExit,'strength must be'):
                runtime.resolve_loras(SimpleNamespace(prompt='x',lora=['x'],cfg=None,negative=None),models.profile('realvis'),'',lambda _:None)


if __name__ == '__main__':
    unittest.main()
