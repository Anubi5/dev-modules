"""Pinned helper validation is offline and preserves conflicting model data."""
import hashlib
import importlib.util
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import helper_models


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


if __name__ == '__main__': unittest.main()
