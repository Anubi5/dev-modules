"""Pinned helper validation is offline and preserves conflicting model data."""
import hashlib
import importlib.util
import contextlib
import json
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import helper_models
import gen


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root/'models'; self.models.mkdir()
        self.legacy = self.root/'legacy'; self.legacy.mkdir()
        self.manifest = dict(schema='local-ai.helper-models/1', files=[])
        for name in ['pose.onnx', 'face.onnx']:
            data = ('fixture '+name).encode()
            (self.models/name).write_bytes(data); (self.legacy/name).write_bytes(data)
            self.manifest['files'].append(dict(path=name, legacy=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))

    def test_verify_is_read_only_and_does_not_touch_legacy_weights(self):
        result = helper_models.verify(self.models, self.manifest)
        self.assertEqual(result['files'], 2)
        self.assertEqual(len(list(self.legacy.iterdir())), 2)
        self.assertEqual(result['retired_duplicates'], [])

    def test_missing_or_changed_models_stop_without_download_or_replacement(self):
        (self.models/'pose.onnx').write_bytes(b'changed')
        with self.assertRaises(ValueError): helper_models.verify(self.models, self.manifest)
        self.assertEqual((self.models/'pose.onnx').read_bytes(), b'changed')
        self.assertEqual(len(list(self.legacy.iterdir())), 2)

    def test_conflict_preserves_every_legacy_file_even_after_an_earlier_match(self):
        (self.legacy/'face.onnx').write_bytes(b'conflict')
        with self.assertRaisesRegex(ValueError, 'differs'):
            helper_models.verify(self.models, self.manifest, self.legacy)
        self.assertEqual(len(list(self.legacy.iterdir())), 2)

    def test_explicit_future_update_retires_only_verified_legacy_duplicates(self):
        result = helper_models.verify(self.models, self.manifest, self.legacy)
        self.assertEqual(len(result['retired_duplicates']), 2)
        self.assertEqual(len(list(self.models.iterdir())), 2)
        self.assertEqual(list(self.legacy.iterdir()), [])

    def test_permanent_store_cannot_be_used_as_the_retirement_directory(self):
        with self.assertRaisesRegex(ValueError, 'permanent model store'):
            helper_models.verify(self.models, self.manifest, self.models)
        self.assertEqual(len(list(self.models.iterdir())), 2)

    def test_verify_only_model_check_never_downloads_moves_or_quarantines(self):
        def forbidden(*args, **kwargs): raise AssertionError('Network operation attempted')
        stub = SimpleNamespace(HfApi=forbidden, hf_hub_download=forbidden)
        spec = importlib.util.spec_from_file_location('offline_fetch_models', Path(__file__).with_name('fetch_models.py'))
        with patch.dict(sys.modules, huggingface_hub=stub):
            fetch = importlib.util.module_from_spec(spec); spec.loader.exec_module(fetch)
        listing=self.root/'models.txt'; listing.write_text('fixture/repo weights.ckpt . sha256:'+'0'*64+'\n')
        old=self.legacy/'weights.ckpt'; old.write_bytes(b'original')
        on_d=self.models/'weights.ckpt'; on_d.write_bytes(b'conflicting')
        state=self.root/'state'
        with patch.object(sys,'argv',['fetch_models.py','--verify-only',str(listing),str(self.models),str(self.models),str(state),str(self.legacy)]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fetch.main(),1)
        self.assertEqual(old.read_bytes(),b'original')
        self.assertEqual(on_d.read_bytes(),b'conflicting')
        self.assertFalse((self.models/'weights.ckpt.quarantined').exists())

    def test_optional_absence_is_reported_at_install_but_explicit_verify_is_strict(self):
        (self.models/'pose.onnx').unlink()
        result = helper_models.verify(self.models, self.manifest, allow_missing=True)
        self.assertEqual(result['missing_optional'], ['pose.onnx'])
        with self.assertRaises(ValueError): helper_models.verify(self.models, self.manifest)
        (self.models/'face.onnx').write_bytes(b'bad')
        with self.assertRaises(ValueError): helper_models.verify(self.models, self.manifest, allow_missing=True)

    def test_capability_checks_are_size_only_and_unrelated_workflows_need_no_helpers(self):
        row = self.manifest['files'][0]
        row['path'] = 'controlnet_aux/pose.onnx'
        (self.models/'controlnet_aux').mkdir()
        (self.models/row['path']).write_bytes(b'x'*row['bytes'])
        with patch.object(helper_models, 'digest', side_effect=AssertionError('Expensive hash at use time')):
            helper_models.check_workflow(self.models, {'1':dict(class_type='DWPreprocessor')}, self.manifest)
            helper_models.check_workflow(self.root/'absent', {'1':dict(class_type='KSampler')})
        (self.models/row['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'pose needs'):
            helper_models.check_workflow(self.models, {'1':dict(class_type='DWPreprocessor')}, self.manifest)
        with self.assertRaisesRegex(ValueError, 'FaceID needs'):
            helper_models.check_workflow(self.root/'absent', {'1':dict(class_type='IPAdapterUnifiedLoaderFaceID')})

    def test_missing_capability_stops_before_service_or_gpu_changes(self):
        with patch.object(gen, 'MODELS', str(self.root/'absent')), patch.object(gen, 'comfy_up', side_effect=AssertionError('Service touched')):
            self.assertIn('pose needs', gen.prepare(print, 1, {'1':dict(class_type='DWPreprocessor')}))

    def test_service_startup_has_no_helper_hash_gate_and_stays_offline(self):
        service=Path(__file__).with_name('comfyui.service').read_text()
        self.assertNotIn('ExecStartPre=', service)
        self.assertIn('HF_HUB_OFFLINE=1', service)
        self.assertIn('--offline', service)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.ro=self.root/'models'; self.ro.mkdir()
        self.state=self.root/'state'; self.listing=self.root/'models.txt'
        self.data=b'pinned model fixture'; self.sha=hashlib.sha256(self.data).hexdigest()
        stub=SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(list_repo_files=lambda repo:['weights.safetensors'],
                             model_info=lambda *a,**k:SimpleNamespace(gated=False,siblings=[SimpleNamespace(rfilename='weights.safetensors',size=len(self.data))])),
                             hf_hub_download=self.download)
        spec=importlib.util.spec_from_file_location('fixture_fetch', Path(__file__).with_name('fetch_models.py'))
        with patch.dict(sys.modules, huggingface_hub=stub):
            self.fetch=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.fetch)
        self.fetch.HfApi=stub.HfApi;self.fetch.hf_hub_download=stub.hf_hub_download
        self.calls=[]

    def download(self, repo, filename, **kwargs):
        self.calls.append((repo, filename, kwargs))
        path=Path(kwargs['local_dir'])/filename; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.data); return str(path)

    def run_fetch(self, flags=(), repo='Bingsu/adetailer', pin=True):
        self.listing.write_text(f'{repo} weights.safetensors checkpoints'+(' sha256:'+self.sha if pin else '')+'\n')
        with patch.object(sys,'argv',['fetch',*flags,str(self.listing),str(self.ro),str(self.ro),str(self.state)]), \
             patch.dict(os.environ,DEV_MODEL_OPERATION='approved',DEV_UNATTENDED='',DEV_APPROVED_SHA256=json.dumps([self.sha])), \
             contextlib.redirect_stdout(io.StringIO()):
            return self.fetch.main()

    def test_no_implicit_download_even_with_writable_destination(self):
        self.assertEqual(self.run_fetch(),1); self.assertEqual(self.calls,[])

    def test_explicit_approved_pinned_download_and_existing_file_is_preserved(self):
        self.assertEqual(self.run_fetch(['--allow-downloads']),0)
        self.assertEqual(len(self.calls),1); self.assertIs(self.calls[0][2]['token'],False)
        self.assertEqual(self.run_fetch(['--allow-downloads']),0)
        self.assertEqual(len(self.calls),1)
        self.assertEqual((self.ro/'checkpoints/weights.safetensors').read_bytes(),self.data)

    def test_unpinned_unknown_and_restricted_origins_never_reach_network(self):
        self.assertEqual(self.run_fetch(['--allow-downloads'],pin=False),1)
        for repo in ['unknown/repo','h94/IP-Adapter-FaceID','black-forest-labs/flux-3-image-klein-comfy']:
            self.assertEqual(self.run_fetch(['--allow-downloads'],repo=repo),1)
        self.assertEqual(self.calls,[])

    def test_wrong_download_hash_is_never_published(self):
        self.sha='0'*64
        self.assertEqual(self.run_fetch(['--allow-downloads']),1)
        self.assertFalse((self.ro/'checkpoints/weights.safetensors').exists())

    def test_existing_corrupt_file_is_not_quarantined_or_overwritten(self):
        path=self.ro/'checkpoints/weights.safetensors'; path.parent.mkdir(); path.write_bytes(b'corrupt')
        self.assertEqual(self.run_fetch(['--allow-downloads']),1)
        self.assertEqual(path.read_bytes(),b'corrupt'); self.assertEqual(self.calls,[])
        self.assertFalse(path.with_suffix('.safetensors.quarantined').exists())

    def test_explicit_verification_rehashes_even_when_size_and_timestamp_match(self):
        self.assertEqual(self.run_fetch(['--allow-downloads']),0)
        path=self.ro/'checkpoints/weights.safetensors'; stat=path.stat()
        path.write_bytes(b'x'*len(self.data)); os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        self.assertEqual(self.run_fetch(['--verify-only']),1)
        self.assertEqual(path.read_bytes(),b'x'*len(self.data))

    def test_update_reuses_the_existing_unchanged_hash_record(self):
        self.assertEqual(self.run_fetch(['--allow-downloads']),0)
        with patch.object(self.fetch,'sha256',side_effect=AssertionError('Unchanged model rehashed during update')):
            self.assertEqual(self.run_fetch(['--allow-downloads']),0)

    def test_every_declared_model_has_a_valid_pin(self):
        rows=[line.split() for line in Path(__file__).with_name('models.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(any(part.startswith('sha256:') and self.fetch.re.fullmatch('[0-9a-fA-F]{64}',part[7:]) for part in row))


if __name__ == '__main__': unittest.main()
