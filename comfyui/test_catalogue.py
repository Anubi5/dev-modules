"""Local discovery never downloads, executes discovered code or modifies weights."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import catalogue
from generation import registry


class CatalogueTests(unittest.TestCase):
    def test_default_text_hides_components_but_preserves_unknown_and_missing_packages(self):
        package=dict(id='fixture',status='needs-something',components=['comfy/component.gguf'])
        files=[dict(id=name,path=str(self.root/name),status='untested',expected_sha256=pin)
               for name,pin in [('comfy/component.gguf','1'*64),('standalone-pinned.gguf','2'*64),('unrecognized.gguf',None)]]
        report=dict(schema='local-ai.models/1',packages=[package],files=files)
        with patch.object(catalogue,'inventory',return_value=report):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(catalogue.main(['list','--root',str(self.root)]),0)
            self.assertIn('fixture',output.getvalue())
            self.assertIn('unrecognized.gguf',output.getvalue())
            self.assertIn('standalone-pinned.gguf',output.getvalue())
            self.assertNotIn('comfy/component.gguf',output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(catalogue.main(['list','--all','--root',str(self.root)]),0)
            self.assertIn('comfy/component.gguf',output.getvalue())
            self.assertIn('standalone-pinned.gguf',output.getvalue())
            self.assertIn('unrecognized.gguf',output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(catalogue.main(['list','--json']),0)
            self.assertEqual(json.loads(output.getvalue()),report)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(catalogue.main(['list','--all','--json']),0)
            self.assertEqual(json.loads(output.getvalue()),report)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'models';self.root.mkdir()
        self.data=b'fixture model data';self.sha=hashlib.sha256(self.data).hexdigest()

    def fixture(self,relative):
        p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(self.data);return p

    def test_copied_gguf_lora_and_unknown_checkpoint_are_visible_without_registration(self):
        for name in ['llm/new.gguf','comfy/loras/new.safetensors','comfy/diffusion_models/new.safetensors']:
            self.fixture(name)
        self.fixture('do_not_execute.py').write_text("raise AssertionError('Executed discovered code')")
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report=catalogue.inventory(self.root)
        self.assertEqual(catalogue.select(report,'new.gguf')['status'],'untested')
        self.assertEqual(catalogue.select(report,'comfy/loras/new.safetensors')['status'],'untested')
        self.assertEqual(catalogue.select(report,'comfy/diffusion_models/new.safetensors')['status'],'needs-something')
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_package_reports_missing_parts_then_ready_without_loading_them(self):
        package=dict(name='fixture',kind='image',architecture='fixture',sha256='1'*64,
                     parts={'model':dict(folder='diffusion_models',file='fixture.safetensors',bytes=len(self.data),sha256=self.sha)})
        with patch.object(registry,'packages',return_value={'fixture':package}):
            self.assertEqual(catalogue.select(catalogue.inventory(self.root),'fixture')['status'],'needs-something')
            self.fixture('comfy/diffusion_models/fixture.safetensors')
            report=catalogue.inventory(self.root)
            self.assertEqual(catalogue.select(report,'fixture')['status'],'ready')
            self.assertEqual(catalogue.verify(self.root,report,'fixture')['files'][0]['sha256'],self.sha)

    def test_hash_mismatch_preserves_file_and_fails_verification(self):
        p=self.fixture('bad.gguf')
        with patch.object(catalogue,'declared_pins',return_value={'bad.gguf':'0'*64}):
            report=catalogue.inventory(self.root)
            with self.assertRaisesRegex(ValueError,'mismatch'): catalogue.verify(self.root,report,'bad.gguf')
        self.assertEqual(p.read_bytes(),self.data)

    def test_ambiguous_names_are_rejected(self):
        self.fixture('a/model.gguf');self.fixture('b/model.gguf')
        with self.assertRaises(ValueError): catalogue.select(catalogue.inventory(self.root),'model.gguf')

    def test_symlink_escapes_are_rejected(self):
        outside=Path(self.temp.name)/'outside.gguf';outside.write_bytes(self.data)
        try: (self.root/'escape.gguf').symlink_to(outside)
        except OSError as error:
            if getattr(error,'winerror',None)==1314: self.skipTest('Windows account cannot create symlinks; Linux CI must run this check')
            raise
        report=catalogue.inventory(self.root)
        self.assertNotIn('escape.gguf',[r['id'] for r in report['files']])

    def test_serialization_exposes_no_executable_builder_and_keeps_quality_separate(self):
        self.fixture('new.gguf');report=catalogue.inventory(self.root)
        self.assertTrue(report['discovery_read_only'])
        self.assertNotIn('builder.py',json.dumps(report))
        result=catalogue.verify(self.root,report,'new.gguf')
        self.assertFalse(result['files'][0]['pinned'])
        self.assertIn('not GPU',result['meaning'])


if __name__=='__main__':unittest.main()
