"""Separate serving contexts share one canonical weight file; no alias download or copy."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import make_config


def forbidden(*args,**kwargs):raise AssertionError('Unexpected model network operation')
stub=SimpleNamespace(HfApi=lambda:SimpleNamespace(list_repo_files=forbidden,model_info=forbidden),hf_hub_download=forbidden)
spec=importlib.util.spec_from_file_location('variant_resolver',Path(__file__).with_name('resolve_models.py'))
with patch.dict(sys.modules,huggingface_hub=stub):
    resolver=importlib.util.module_from_spec(spec);spec.loader.exec_module(resolver)


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.ro=self.root/'ro';self.state=self.root/'state'
        self.file=self.ro/'daily/model-Q6_K.gguf';self.file.parent.mkdir(parents=True);self.file.write_bytes(b'canonical weights fixture')
        self.state.mkdir();self.base=dict(id='daily',name='daily',repo='fixture/repo',quant=['Q6_K'],args='-c 32768',alias='default',vision=True)
        self.agent=dict(id='agent',name='agent',weights_from='daily',args='-c 65536')
        self.record=dict(repo='fixture/repo',files=[str(self.file)],file=str(self.file),quant=['Q6_K'])
        (self.state/'models.json').write_text(json.dumps({'daily':self.record}))

    def resolve(self,specs):
        with contextlib.redirect_stdout(io.StringIO()):return resolver.chat_models(specs,str(self.state),str(self.ro))

    def test_variant_reuses_exact_paths_without_copy_download_or_order_dependency(self):
        before=self.file.read_bytes()
        self.assertEqual(self.resolve([self.agent,self.base]),0)
        records=json.loads((self.state/'models.json').read_text())
        self.assertEqual(records['daily']['files'],records['agent']['files'])
        self.assertEqual(records['agent']['weights_from'],'daily')
        self.assertFalse((self.ro/'agent').exists())
        self.assertEqual(list(self.ro.rglob('*.gguf')),[self.file]);self.assertEqual(self.file.read_bytes(),before)

    def test_generated_contexts_are_separate_and_default_alias_is_not_duplicated(self):
        self.assertEqual(self.resolve([self.base,self.agent]),0)
        records=json.loads((self.state/'models.json').read_text())
        with patch.object(make_config,'AI',str(self.root)):
            text=make_config.build({'chat':[self.base,self.agent]},records)
        self.assertIn('-c 32768',text);self.assertIn('-c 65536',text)
        self.assertEqual(text.count('-m '+str(self.file)),2)
        self.assertEqual(text.count('- "default"'),1)

    def test_missing_reference_and_alias_chains_fail_without_network(self):
        self.assertNotEqual(self.resolve([dict(self.agent,weights_from='missing')]),0)
        self.assertNotEqual(self.resolve([self.base,self.agent,dict(id='chain',name='chain',weights_from='agent')]),0)

    def test_missing_canonical_source_fails_without_creating_an_alias_directory(self):
        self.file.unlink()
        self.assertNotEqual(self.resolve([self.base,self.agent]),0)
        self.assertFalse((self.ro/'agent').exists())

    def test_duplicate_ids_fail_without_changing_the_existing_record(self):
        old=(self.state/'models.json').read_bytes()
        self.assertNotEqual(self.resolve([self.base,self.base]),0)
        self.assertEqual((self.state/'models.json').read_bytes(),old)


if __name__=='__main__':unittest.main()
