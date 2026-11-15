"""Fresh-state local adoption rejects ambiguous/incomplete sets without touching weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resolve_models as resolver


class LocalAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'ro' / 'daily'
        self.folder.mkdir(parents=True)
        self.spec = dict(id='daily', quant=['Q6_K', 'Q4_K_M'])

    def files(self, names):
        for name in names:
            (self.folder / name).write_bytes(b'local fixture')

    def adopt(self):
        with patch.object(resolver, '_hf_download', side_effect=AssertionError('Unexpected download')):
            return resolver.adopt_local(self.spec, str(self.folder.parent))

    def test_complete_split_is_ordered_and_manifest_uses_existing_paths(self):
        names = ['model-Q6_K-00002-of-00002.gguf', 'model-Q6_K-00001-of-00002.gguf']
        self.files(names)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        state = self.root / 'state'
        with patch.object(resolver, '_hf_download', side_effect=AssertionError('Unexpected download')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(resolver.chat_models([self.spec], str(state), str(self.folder.parent)), 0)
        entry = json.loads((state / 'models.json').read_text())['daily']
        self.assertEqual(entry['files'], [str(self.folder / n) for n in sorted(names)])
        self.assertEqual(entry['file'], entry['files'][0])
        self.assertIn('not hash-pinned', output.getvalue())
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()})

    def test_invalid_split_and_ambiguous_sets_are_rejected(self):
        cases = [
            ['model-Q6_K-00001-of-00002.gguf'],
            ['model-Q6_K-00002-of-00002.gguf'],
            ['model-Q6_K-00001-of-00002.gguf', 'model-Q6_K-00003-of-00002.gguf'],
            ['model-Q6_K-00000-of-00002.gguf', 'model-Q6_K-00002-of-00002.gguf'],
            ['model-Q6_K-00001-of-00002.gguf', 'model-Q6_K-00002-of-00003.gguf'],
            ['model-Q6_K-00001-of-00002.gguf', 'other-Q6_K-00002-of-00002.gguf'],
            ['model-Q6_K-00001-of-00002.gguf', 'model-Q6_K-00001-of-00002.GGUF'],
            ['model-Q6_K.gguf', 'other-Q6_K.gguf'],
            ['model-Q6_K.gguf', 'model-Q6_K-00001-of-00001.gguf'],
        ]
        for names in cases:
            with self.subTest(names=names):
                for p in self.folder.iterdir(): p.unlink()
                self.files(names)
                self.assertIsNone(self.adopt())
                self.assertEqual(sorted(p.name for p in self.folder.iterdir()), sorted(names))

    def test_preferred_quant_and_projector_are_retained(self):
        self.files(['model-Q6_K.gguf', 'model-Q4_K_M.gguf', 'mmproj.gguf'])
        self.spec['mmproj'] = True
        entry = self.adopt()
        self.assertEqual(entry['files'], [str(self.folder / 'model-Q6_K.gguf')])
        self.assertEqual(entry['mmproj'], str(self.folder / 'mmproj.gguf'))

    def test_required_projector_missing_empty_or_ambiguous_is_rejected(self):
        self.files(['model-Q6_K.gguf'])
        self.spec['mmproj'] = True
        self.assertIsNone(self.adopt())
        (self.folder / 'mmproj.gguf').touch()
        self.assertIsNone(self.adopt())
        self.files(['mmproj.gguf', 'other-mmproj.gguf'])
        self.assertIsNone(self.adopt())

    def test_empty_model_is_rejected(self):
        (self.folder / 'model-Q6_K.gguf').touch()
        self.assertIsNone(self.adopt())


if __name__ == '__main__':
    unittest.main()