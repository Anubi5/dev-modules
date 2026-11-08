"""Offline hash approval and publication boundaries; no real network."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
hub=ModuleType('huggingface_hub');hub.HfApi=lambda **kwargs:SimpleNamespace()
hub.hf_hub_download=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected network'))
with patch.dict(sys.modules,huggingface_hub=hub):
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,ROOT/path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    comfy=load('fixture_comfy_fetch','comfyui/fetch_models.py')
    llama=load('fixture_llama_resolve','llama-swap/resolve_models.py')
    rig=load('fixture_rig_assets','rig/fetch_assets.py')


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'weights';self.root.mkdir()
        self.data=b'fixture pinned weight';self.sha=hashlib.sha256(self.data).hexdigest()
        self.row=dict(file='probe.gguf',path='probe.gguf',bytes=len(self.data),sha256=self.sha)
        self.spec=dict(revision='1'*40,download_policy=dict(approved=True,license='fixture-licence'))
        env=patch.dict(os.environ,DEV_MODEL_OPERATION='approved',DEV_APPROVED_SHA256=json.dumps([self.sha]),DEV_UNATTENDED='')
        env.start();self.addCleanup(env.stop)

    def snapshot(self):return {p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def comfy_run(self,race=False):
        name='probe.safetensors';listing=self.base/'models.txt'
        listing.write_text(f'Bingsu/adetailer {name} checkpoints sha256:{self.sha}\n')
        def fetch(*args,**kwargs):
            path=Path(kwargs['local_dir'])/name;path.write_bytes(self.data)
            if race:
                dest=self.root/'checkpoints'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(b'owner data')
            return str(path)
        argv=['fetch','--allow-downloads',str(listing),str(self.root),str(self.root),str(self.base/'state')]
        with patch.object(sys,'argv',argv),patch.object(comfy,'HfApi',return_value=SimpleNamespace(list_repo_files=lambda *a:[name])) as metadata, \
             patch.object(comfy,'hf_hub_download',side_effect=fetch) as download,contextlib.redirect_stdout(io.StringIO()):
            code=comfy.main()
        return code,download.call_count,metadata.call_count

    def test_every_fetcher_requires_an_approved_hash(self):
        for module in (comfy,llama,rig):
            self.assertTrue(module.approved_download(self.sha));self.assertFalse(module.approved_download('0'*64))
            for value in ('[]','bad','{}','[null]',json.dumps([self.sha,'bad'])):
                with patch.dict(os.environ,DEV_APPROVED_SHA256=value):self.assertFalse(module.approved_download(self.sha))

    def test_unattended_and_verification_ignore_approvals(self):
        for module in (comfy,llama,rig):
            with patch.dict(os.environ,DEV_UNATTENDED='1'):self.assertFalse(module.approved_download(self.sha))
            with patch.dict(os.environ,DEV_MODEL_OPERATION='verify'):self.assertFalse(module.approved_download(self.sha))

    def test_unapproved_comfy_never_queries_or_downloads(self):
        with patch.dict(os.environ,DEV_APPROVED_SHA256='[]'):self.assertEqual(self.comfy_run(),(1,0,0))
        self.assertEqual(self.snapshot(),{})

    def test_approved_comfy_and_staging_cleanup(self):
        self.assertEqual(self.comfy_run(),(0,1,1));self.assertEqual(self.snapshot(),{'checkpoints/probe.safetensors':self.data})

    def test_comfy_racing_destination_and_failed_staging_preserved(self):
        self.assertEqual(self.comfy_run(race=True)[0],1)
        self.assertEqual((self.root/'checkpoints/probe.safetensors').read_bytes(),b'owner data')
        self.assertTrue(any(p.is_file() for p in self.root.glob('.model-stage-*/probe.safetensors')))

    def test_llama_unapproved_refused_before_staging(self):
        with patch.dict(os.environ,DEV_APPROVED_SHA256='[]'):
            with self.assertRaises(RuntimeError):llama.place('fixture/probe.gguf',str(self.root),str(self.root),str(self.base/'legacy'),'repo',declaration=(self.spec,self.row))
        self.assertEqual(self.snapshot(),{})

    def test_llama_approved_pins_and_bytes_verified(self):
        def fetch(repo,filename,**kwargs):
            self.assertEqual(kwargs['revision'],self.spec['revision']);path=Path(kwargs['local_dir'])/filename;path.write_bytes(self.data);return str(path)
        with patch.object(llama,'download',side_effect=fetch):llama.place('fixture/probe.gguf',str(self.root),str(self.root),str(self.base/'legacy'),'repo',declaration=(self.spec,self.row))
        self.assertEqual(self.snapshot(),{'fixture/probe.gguf':self.data})

    def test_llama_pin_mismatch_never_repaired(self):
        folder=self.root/'fixture';folder.mkdir();weight=folder/'probe.gguf';weight.write_bytes(b'owner data')
        state=self.base/'state';state.mkdir()
        spec=dict(self.spec,id='fixture',repo='repo',quant=['Q4_K_M'],pinned_files=[self.row])
        self.assertEqual(llama.chat_models([spec],str(state),str(self.root),str(self.root)),1)
        self.assertEqual(weight.read_bytes(),b'owner data')

    def test_rig_unapproved_or_gated_refused(self):
        with patch.dict(os.environ,DEV_APPROVED_SHA256='[]'):
            with self.assertRaises(SystemExit):rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',self.spec['download_policy'])
        with self.assertRaises(SystemExit):rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',dict(approved=True,license='fixture',gated=True))
        self.assertEqual(self.snapshot(),{})

    def test_rig_approved_and_no_clobber(self):
        def fetch(**kwargs):
            path=Path(kwargs['local_dir'])/self.row['path'];path.write_bytes(self.data);return str(path)
        module=ModuleType('huggingface_hub');module.hf_hub_download=fetch
        with patch.dict(sys.modules,huggingface_hub=module):
            rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',self.spec['download_policy'])
            with self.assertRaises(FileExistsError):rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',self.spec['download_policy'])
        self.assertEqual((self.root/'probe.gguf').read_bytes(),self.data)


if __name__=='__main__':unittest.main()
