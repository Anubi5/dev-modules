"""Module commands answer their own --help (the platform forwards it): no inventory, GPU, process or network use."""
import contextlib
import importlib.util
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'comfyui'))
spec=importlib.util.spec_from_file_location('gen_help_subject',ROOT/'comfyui/gen.py')
gen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def real_help(script, arguments):
    """Run a module CLI as the platform would with help arguments; return what it printed."""
    old_path, old_argv = sys.path[:], sys.argv[:]
    try:
        sys.path.insert(0, str(script.parent)); sys.argv = [str(script), *arguments]
        with patch('subprocess.Popen', side_effect=AssertionError('Help started a process')),              patch('socket.socket', side_effect=AssertionError('Help accessed network')),              contextlib.redirect_stdout(io.StringIO()) as output:
            try:
                runpy.run_path(str(script), run_name='__main__')
            except SystemExit as result:
                assert result.code in (0, None), result.code
        return output.getvalue()
    finally:
        sys.path[:], sys.argv[:] = old_path, old_argv


class HelpTest(unittest.TestCase):
    def test_character_help_is_the_selected_subcommand(self):
        for name,option in [('manifest','--file'),('inspect','manifest'),('plan','pipeline'),('run','--work-dir')]:
            with self.subTest(command=name):
                text=real_help(ROOT/'character/cli.py',[name,'--help'])
                self.assertIn('ai character '+name,text)
                self.assertIn(option,text)

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
