"""Prepared-runtime installer reuse with fake tools, no downloads or GPU (Linux CI)."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform.startswith('linux'), 'installer uses Linux shell paths')
class OfflineInstallTests(unittest.TestCase):
    def test_repeat_install_skips_resolution_and_changed_pin_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mod=root/'comfyui';mod.mkdir();home=root/'home';home.mkdir()
            for name in ('install.sh','nodes.txt'):
                shutil.copyfile(HERE/name,mod/name)
            # This fixture tests dependency reuse; catalogue migration has its own contract tests.
            (mod/'catalogue.py').write_text('def migrate_acceptance():\n    pass\n')
            app=mod/'app/ComfyUI';(app/'.git').mkdir(parents=True)
            ref=re.search(r'^COMFY_REF=([a-f0-9]+)',(mod/'install.sh').read_text(),re.M).group(1)
            (app/'.eva-ref').write_text(ref)
            for line in (mod/'nodes.txt').read_text().splitlines():
                if not line.strip() or line.lstrip().startswith('#'):continue
                url,pin,*_=line.split();node=app/'custom_nodes'/url.rsplit('/',1)[-1].removesuffix('.git')
                (node/'.git').mkdir(parents=True);(node/'.eva-ref').write_text(pin)
                (node/'requirements.txt').write_text('git+https://github.com/facebookresearch/sam2\n' if node.name=='ComfyUI-Impact-Pack' else 'fixture-package\n')
            python=app/'.venv/bin/python';python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nexit 0\n');python.chmod(0o700)
            fake=root/'bin';fake.mkdir();log=root/'resolver.log'
            (fake/'git').write_text('#!/bin/sh\n[ "$1" = -C ] && [ "$3" = rev-parse ] && { cat "$2/.eva-ref"; exit; }\necho unexpected-git >&2; exit 91\n')
            (fake/'uv').write_text('#!/bin/sh\nif [ "$2" = freeze ]; then echo torch==fixture; exit 0; fi\necho resolve >> "$RESOLUTION_LOG"\n[ -n "$REFUSE_RESOLUTION" ] && exit 92\nexit 0\n')
            for tool in fake.iterdir():tool.chmod(0o700)
            env=dict(os.environ,HOME=str(home),MOD=str(mod),PATH=str(fake)+os.pathsep+os.environ['PATH'],
                     PYTHONPATH=str(HERE),MODELS_MOUNT=str(root/'models'),RESOLUTION_LOG=str(log))
            first=subprocess.run(['bash',str(mod/'install.sh')],env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            self.assertTrue(log.exists())
            log.unlink()
            second=subprocess.run(['bash',str(mod/'install.sh')],env=dict(env,REFUSE_RESOLUTION='1'),capture_output=True,text=True,timeout=20)
            self.assertEqual(second.returncode,0,second.stdout+second.stderr)
            self.assertIn('runtime unchanged',second.stdout)
            self.assertFalse(log.exists())
            # A changed declared source pin cannot inherit the prepared-runtime stamp.
            with (mod/'nodes.txt').open('a') as stream:stream.write('\n# changed reviewed node declaration\n')
            changed=subprocess.run(['bash',str(mod/'install.sh')],env=dict(env,REFUSE_RESOLUTION='1'),capture_output=True,text=True,timeout=20)
            self.assertNotEqual(changed.returncode,0)
            self.assertTrue(log.exists())


if __name__=='__main__':unittest.main()
