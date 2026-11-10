"""Actual generation dispatch must print help without inventory or GPU calls."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'comfyui'))
spec=importlib.util.spec_from_file_location('gen_help_subject',ROOT/'comfyui/gen.py')
gen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


class HelpTest(unittest.TestCase):
    def test_documented_generation_example_uses_real_parser(self):
        args=gen.generation_parser().parse_intermixed_args(['image','a blue mug','--model','krea2','--seed','43','--json'])
        self.assertEqual((args.kind,args.model,args.seed,args.json),('image','krea2',43,True))

    def test_actual_dispatch_has_no_effects(self):
        for arguments in [['--help'],['-h'],['help'],['schema','--help'],['schema','-h'],['image','--help'],['models','--help'],['jobs','--help'],['list','--help']]:
            with self.subTest(arguments=arguments), patch.object(sys,'argv',['ai gen',*arguments]), \
                 patch.object(gen,'models_report',side_effect=AssertionError('Help scanned models')), \
                 patch.object(gen,'dump_schema',side_effect=AssertionError('Help wrote schema')), \
                 patch('subprocess.Popen',side_effect=AssertionError('Help started process')), \
                 patch('socket.socket',side_effect=AssertionError('Help accessed network')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                try:
                    self.assertEqual(gen.main(),0)
                except SystemExit as result:
                    self.assertEqual(result.code,0)
                self.assertIn('usage:',output.getvalue())


if __name__=='__main__': unittest.main()
