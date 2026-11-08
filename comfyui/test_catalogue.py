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
    def test_failed_staging_is_visible_with_size_path_and_no_execution(self):
        direct=self.fixture('comfy/checkpoints/model.'+'a'*32+'.part')
        cached=self.fixture('llm/.model-stage-fixture/model.gguf')
        self.fixture('llm/.model-stage-fixture/.cache/huggingface/download/model.metadata')
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report=catalogue.inventory(self.root)
        for path in (direct,cached):
            row=catalogue.select(report,str(path))
            self.assertEqual(row['status'],'failed-download');self.assertFalse(row['runnable'])
            self.assertEqual(row['capability_tags'],[])
        self.assertFalse(any(row['id'].endswith('.metadata') for row in report['files']))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(catalogue.main(['list','--root',str(self.root)]),0)
        self.assertIn('failed download',output.getvalue());self.assertIn(str(len(self.data))+' bytes',output.getvalue())
        self.assertIn(str(direct),output.getvalue());self.assertIn(str(cached),output.getvalue())
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
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

    def test_headers_report_format_without_inventing_capabilities_or_serving_entries(self):
        valid=self.fixture('renamed-image-video-llm.gguf')
        valid.write_bytes(b'GGUF'+(3).to_bytes(4,'little')+bytes(16))
        bad=self.fixture('truncated.gguf');bad.write_bytes(b'GGUF')
        report=catalogue.inventory(self.root)
        row=catalogue.select(report,valid.name)
        self.assertEqual(row['format']['status'],'recognized')
        self.assertEqual(row['capability_tags'],[])
        self.assertEqual(row['identification'],'unknown')
        self.assertFalse(row['runnable'])
        self.assertEqual(catalogue.select(report,bad.name)['format']['completeness'],'invalid')

    def test_declared_new_model_uses_existing_family_without_cli_changes(self):
        header=json.dumps({'weight':{'dtype':'U8','shape':[4],'data_offsets':[0,4]}}).encode()
        path=self.fixture('comfy/diffusion_models/new.safetensors')
        path.write_bytes(len(header).to_bytes(8,'little')+header+b'data')
        package=dict(kind='image',architecture='existing-family',sha256='1'*64,capability_tags=['image'],
                     parts={'model':dict(folder='diffusion_models',file=path.name,bytes=path.stat().st_size,sha256=catalogue.sha256(path))})
        before=path.stat()
        with patch.object(registry,'packages',return_value={'new-variant':package}):
            report=catalogue.inventory(self.root)
        model=catalogue.select(report,'new-variant')
        self.assertEqual(model['capability_tags'],['image'])
        self.assertTrue(model['runnable'])
        self.assertEqual(model['completeness'],'declared-components-present')
        row=catalogue.select(report,path.name)
        self.assertEqual(row['identification'],'declared-package')
        self.assertEqual(row['capability_tags'],['image'])
        self.assertFalse(row['runnable'])
        self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns)
        path.write_bytes(path.read_bytes()[:-1])
        self.assertEqual(catalogue.format_info(path)['status'],'invalid')

    def test_pickle_checkpoints_are_not_deserialized(self):
        path=self.fixture('hostile.pt')
        with patch('pickle.load',side_effect=AssertionError('Checkpoint executed')):
            row=catalogue.select(catalogue.inventory(self.root),path.name)
        self.assertEqual(row['format']['status'],'unverified')
        self.assertFalse(row['runnable'])

    def test_unknown_architecture_does_not_gain_execution_from_complete_files(self):
        source=Path(self.temp.name)/'module';source.mkdir()
        header=json.dumps({'weight':{'data_offsets':[0,4]}}).encode()
        path=self.fixture('comfy/diffusion_models/fixture.safetensors')
        path.write_bytes(len(header).to_bytes(8,'little')+header+b'data')
        (source/'models.txt').write_text('fixture/repo fixture.safetensors diffusion_models sha256:'+catalogue.sha256(path))
        (source/'profiles.json').write_text(json.dumps({'models':{
            'unsupported':{'arch':'new-unreviewed-architecture','file':path.name,'capability_tags':['image']},
            'supported':{'arch':'sdxl','file':path.name,'capability_tags':['image']}}}))
        with patch.object(catalogue,'HERE',source),patch.object(registry,'packages',return_value={}):
            report=catalogue.inventory(self.root)
        self.assertTrue(catalogue.select(report,'supported')['runnable'])
        unknown=catalogue.select(report,'unsupported')
        self.assertFalse(unknown['runnable'])
        self.assertIn('reviewed family',unknown['compatibility'])


if __name__=='__main__':unittest.main()
