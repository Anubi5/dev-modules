"""Module-owned offline planning and shared declaration parsing."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class Plans(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.models=self.base/'weights';self.models.mkdir()

    def module(self,name,filename):
        folder=self.base/name;folder.mkdir();path=folder/filename;shutil.copyfile(ROOT/name/filename,path);return folder,path

    def run_plan(self,path,*args):
        before={str(p):p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        result=subprocess.run([sys.executable,'-B',str(path),'--plan','--json','--models-root',str(self.models),*args],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.base.rglob('*') if p.is_file()})
        return json.loads(result.stdout)['files']

    def test_nonpackage_comfy_pin_with_declared_bytes_is_eligible(self):
        folder,path=self.module('comfyui','fetch_models.py')
        (folder/'models.txt').write_text('fixture/repo upstream.safetensors checkpoints saved.safetensors bytes:34 sha256:'+'a'*64+'\n')
        (folder/'download_licenses.json').write_text(json.dumps({'fixture/repo':dict(approved=True,license='fixture-only')}))
        row,=self.run_plan(path)
        self.assertEqual((row['path'],row['bytes'],row['sha256'],row['eligible']),('comfy/checkpoints/saved.safetensors',34,'a'*64,True))
        spec=importlib.util.spec_from_file_location('own_comfy',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        row,=module.declarations(folder/'models.txt')
        self.assertEqual((row['name'],row['bytes']),('saved.safetensors',34))

    def test_unknown_comfy_bytes_remain_manual_and_invalid_size_refused(self):
        folder,path=self.module('comfyui','fetch_models.py')
        (folder/'models.txt').write_text('fixture/repo weights.safetensors checkpoints sha256:'+'a'*64+'\n')
        (folder/'download_licenses.json').write_text(json.dumps({'fixture/repo':dict(approved=True,license='fixture-only')}))
        row,=self.run_plan(path);self.assertIsNone(row['bytes']);self.assertFalse(row['eligible'])
        with (folder/'models.txt').open('a') as stream:stream.write('fixture/repo bad.safetensors checkpoints bytes:-1\n')
        result=subprocess.run([sys.executable,'-B',str(path),'--plan','--json'],capture_output=True)
        self.assertNotEqual(result.returncode,0)

    def test_llama_owns_plan_and_config_parser(self):
        folder,path=self.module('llama-swap','resolve_models.py')
        row=dict(file='fixture.gguf',bytes=34,sha256='b'*64)
        (folder/'models.json').write_text(json.dumps(dict(chat=[dict(id='fixture',repo='fixture/repo',revision='1'*40,pinned_files=[row],download_policy=dict(approved=True,license='fixture-only'))])))
        item,=self.run_plan(path);self.assertTrue(item['eligible']);self.assertEqual(item['path'],'llm/fixture/fixture.gguf')

    def test_rig_owns_weight_and_architecture_plans(self):
        folder,path=self.module('rig','fetch_assets.py')
        row=dict(path='fixture.ckpt',bytes=34,sha256='c'*64)
        owner=dict(repository='fixture/repo',revision='2'*40,download_policy=dict(approved=True,license='fixture-only'))
        (folder/'models.json').write_text(json.dumps(dict(weights=dict(owner,files=[row]))))
        (folder/'architecture.json').write_text(json.dumps(dict(owner,**dict(row,path='config.json'))))
        files=self.run_plan(path);self.assertEqual(len(files),2);self.assertTrue(all(row['eligible'] for row in files))


if __name__=='__main__':unittest.main()
