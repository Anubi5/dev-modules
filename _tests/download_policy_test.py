"""Offline fixtures: no implicit model writes, prompts precede pinned acquisition, existing data survives."""
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
hub=ModuleType('huggingface_hub')
hub.HfApi=lambda **kwargs:SimpleNamespace()
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
        self.env=patch.dict(os.environ,DEV_MODEL_OPERATION='verify',DEV_UNATTENDED='');self.env.start();self.addCleanup(self.env.stop)

    def snapshot(self):return {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def comfy_run(self,answer='NO',unattended=False,race=False):
        name='probe.safetensors';listing=self.base/'models.txt'
        listing.write_text(f'Bingsu/adetailer {name} checkpoints sha256:{self.sha}\n')
        api=SimpleNamespace(model_info=lambda *a,**k:SimpleNamespace(gated=False,siblings=[SimpleNamespace(rfilename=name,size=len(self.data))]),list_repo_files=lambda *a:[name])
        def fetch(*args,**kwargs):
            path=Path(kwargs['local_dir'])/name;path.write_bytes(self.data)
            if race:
                dest=self.root/'checkpoints'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(b'owner data')
            return str(path)
        argv=['fetch','--allow-downloads',str(listing),str(self.root),str(self.root),str(self.base/'state')]
        with patch.object(sys,'argv',argv),patch.object(comfy,'HfApi',return_value=api) as metadata, \
             patch.object(comfy,'hf_hub_download',side_effect=fetch) as download, \
             patch.object(sys.stdin,'isatty',return_value=True),patch('builtins.input',return_value=answer) as prompt, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code=comfy.main()
        return code,download.call_count,prompt.call_count,metadata.call_count,output.getvalue()

    def test_full_update_and_unattended_never_even_query_remote_metadata(self):
        before=self.snapshot()
        code,downloads,prompts,queries,_=self.comfy_run('DOWNLOAD')
        self.assertEqual((code,downloads,prompts,queries),(1,0,0,0));self.assertEqual(self.snapshot(),before)
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt',DEV_UNATTENDED='1'):
            self.assertEqual(self.comfy_run('DOWNLOAD')[1:4],(0,0,0))

    def test_rejected_download_lists_details_but_never_writes_weights(self):
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'):
            code,downloads,prompts,_,text=self.comfy_run()
        self.assertEqual((code,downloads,prompts),(1,0,1));self.assertEqual(self.snapshot(),{})
        for item in ['probe.safetensors',str(len(self.data)),'Bingsu/adetailer','apache-2.0',self.sha]:self.assertIn(item,text)

    def test_confirmed_comfy_acquisition_and_race_preserve_existing_destination(self):
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'):
            self.assertEqual(self.comfy_run('DOWNLOAD')[:3],(0,1,1))
        self.assertEqual((self.root/'checkpoints/probe.safetensors').read_bytes(),self.data)
        # A second request sees the existing file and does not ask or download.
        self.assertEqual(self.comfy_run('DOWNLOAD')[1:3],(0,0))

    def test_racing_comfy_destination_is_never_overwritten(self):
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'):
            self.assertEqual(self.comfy_run('DOWNLOAD',race=True)[0],1)
        self.assertEqual((self.root/'checkpoints/probe.safetensors').read_bytes(),b'owner data')

    def test_llama_verify_and_declined_acquisition_are_read_only(self):
        with self.assertRaises(RuntimeError):llama.place('fixture/probe.gguf',str(self.root),str(self.root),str(self.base/'legacy'),'repo',declaration=(self.spec,self.row))
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'),patch.object(sys.stdin,'isatty',return_value=True),patch('builtins.input',return_value='NO'):
            with self.assertRaises(RuntimeError):llama.place('fixture/probe.gguf',str(self.root),str(self.root),str(self.base/'legacy'),'repo',declaration=(self.spec,self.row))
        self.assertEqual(self.snapshot(),{})

    def test_llama_pins_and_owner_confirmation_precede_download(self):
        def fetch(repo,filename,**kwargs):
            self.assertEqual(kwargs['revision'],self.spec['revision']);path=Path(kwargs['local_dir'])/filename;path.write_bytes(self.data);return str(path)
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'),patch.object(sys.stdin,'isatty',return_value=True),patch('builtins.input',return_value='DOWNLOAD'),patch.object(llama,'download',side_effect=fetch):
            llama.place('fixture/probe.gguf',str(self.root),str(self.root),str(self.base/'legacy'),'repo',declaration=(self.spec,self.row))
        self.assertEqual((self.root/'fixture/probe.gguf').read_bytes(),self.data)

    def test_full_update_detects_declared_llama_pin_mismatch_without_repair(self):
        folder=self.root/'fixture';folder.mkdir();weight=folder/'probe.gguf';weight.write_bytes(b'owner data')
        state=self.base/'state';state.mkdir()
        spec=dict(self.spec,id='fixture',repo='repo',quant=['Q4_K_M'],pinned_files=[self.row])
        self.assertEqual(llama.chat_models([spec],str(state),str(self.root),str(self.root)),1)
        self.assertEqual(weight.read_bytes(),b'owner data')

    def test_existing_llama_legacy_and_canonical_copies_survive(self):
        legacy=self.base/'legacy';legacy.mkdir();(legacy/'probe.gguf').write_bytes(self.data)
        canonical=self.root/'probe.gguf';canonical.write_bytes(self.data)
        llama.place('probe.gguf',str(self.root),str(self.root),str(legacy))
        self.assertEqual((legacy/'probe.gguf').read_bytes(),self.data)
        canonical.write_bytes(b'different')
        with self.assertRaises(RuntimeError):llama.place('probe.gguf',str(self.root),str(self.root),str(legacy))
        self.assertEqual(canonical.read_bytes(),b'different')

    def test_rig_and_legacy_media_do_not_fetch_missing_unapproved_assets(self):
        with self.assertRaises(SystemExit):rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',self.spec['download_policy'])
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'):
            with self.assertRaises(SystemExit):rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',dict(approved=True,license='fixture',gated=True))
        self.assertEqual(llama.media_models([dict(repo='repo',folder='comfy',file='probe.gguf')],str(self.root)),1)
        self.assertEqual(self.snapshot(),{})

    def test_rig_download_requires_confirmed_pinned_declaration(self):
        def fetch(**kwargs):
            path=Path(kwargs['local_dir'])/self.row['path'];path.write_bytes(self.data);return str(path)
        module=ModuleType('huggingface_hub');module.hf_hub_download=fetch
        with patch.dict(os.environ,DEV_MODEL_OPERATION='prompt'),patch.dict(sys.modules,huggingface_hub=module), \
             patch.object(sys.stdin,'isatty',return_value=True),patch('builtins.input',return_value='DOWNLOAD'):
            rig.acquire('repo','1'*40,self.row,self.root/'probe.gguf',self.spec['download_policy'])
        self.assertEqual((self.root/'probe.gguf').read_bytes(),self.data)


if __name__=='__main__':unittest.main()
