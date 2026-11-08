"""Inspection and nested help must not enter execution or persist caches."""
import contextlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import assets
import gen
from generation import registry


class InspectionTests(unittest.TestCase):
    def test_show_preserves_notes_without_creating_hash_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); models=root/'models'; (models/'loras').mkdir(parents=True)
            model=models/'loras/fixture.safetensors'
            model.write_bytes(struct.pack('<Q',2)+b'{}')
            cache=root/'cache/hashes.json'
            with patch.object(assets,'ROOTS',[str(models)]), patch.object(assets,'CACHE',str(cache)), \
                 patch.object(sys,'argv',['gen.py','show','fixture.safetensors','--json']), \
                 patch.object(assets,'_save',side_effect=AssertionError('Inspection wrote state')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(gen.main(),0)
            self.assertIn('fixture.safetensors',output.getvalue())
            self.assertFalse(cache.exists())
            self.assertEqual(model.read_bytes(),struct.pack('<Q',2)+b'{}')

    def test_nested_help_and_unknown_options_do_not_enter_execution(self):
        for args,code in [(['jobs','--help'],0),(['jobs','--bad'],2),(['jobs','0'],2),
                          (['models','--help'],0),(['models','--bad'],2),(['note','help'],0),
                          (['image','--help'],0),(['rerun','--help'],0)]:
            with self.subTest(args=args), patch.object(sys,'argv',['gen.py']+args), \
                 patch.object(gen,'prepare',side_effect=AssertionError('GPU preparation')), \
                 patch.object(gen,'models_report',side_effect=AssertionError('Unexpected inventory')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped: gen.main()
                self.assertEqual(stopped.exception.code,code)

    def test_krea_description_reaches_generation_profile(self):
        self.assertIn('Krea 2',registry.profiles()['krea2']['about'])


if __name__=='__main__': unittest.main()
