"""Public-skill references and template destination safety, using disposable owner homes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('skills_check',ROOT/'_tests/skills_check.py')
skills=importlib.util.module_from_spec(spec);spec.loader.exec_module(skills)


class ContractSafetyTests(unittest.TestCase):
    def test_nonexistent_public_skill_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            skill=Path(folder);(skill/'SKILL.md').write_text('Run `ai nonexistent do-it`.')
            self.assertIn('missing public command',skills.unknown_commands(skill,{})[0])
            (skill/'SKILL.md').write_text('Inspect `ai mod list`.')
            self.assertEqual(skills.unknown_commands(skill,{}),[])

    def test_template_names_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);env=dict(os.environ,HOME=str(root),USERPROFILE=str(root))
            script=ROOT/'_template/bin/ai-example'
            for name in ('../owner','a/b','a\\b','/tmp/owner','C:owner'):
                with self.subTest(name=name):
                    result=subprocess.run([sys.executable,'-B',str(script),'save',name],env=env,capture_output=True,text=True,timeout=10)
                    self.assertEqual(result.returncode,2,result.stderr)
            result=subprocess.run([sys.executable,'-B',str(script),'save','greeting'],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'work/outputs/example/greeting.txt').read_text(),'hello, greeting\n')

    def test_template_refuses_an_output_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'work/outputs/example';out.mkdir(parents=True)
            owner=root/'owner.txt';owner.write_text('preserve')
            try:(out/'greeting.txt').symlink_to(owner)
            except OSError:self.skipTest('Local symlink privilege unavailable; Linux CI exercises this case')
            result=subprocess.run([sys.executable,'-B',str(ROOT/'_template/bin/ai-example'),'save','greeting'],
                env=dict(os.environ,HOME=str(root),USERPROFILE=str(root)),capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2)
            self.assertEqual(owner.read_text(),'preserve')


if __name__=='__main__':unittest.main()
