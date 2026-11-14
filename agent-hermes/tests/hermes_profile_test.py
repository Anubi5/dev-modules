"""Offline policy checks query no authentication fields and never launch an agent."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'hermes_profile_check.py'
import sys
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('profile_check',SOURCE)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.soul=Path(self.temp.name)/'SOUL.md'
        self.soul.write_text('Existing user personality\n'+SOURCE.with_name('hermes-local-SOUL.md').read_text())
        self.values=dict(checker.EXPECTED,**{'skills.disabled':sorted(checker.DISABLED_SKILLS)})
        self.values.update({key:None for key in checker.FALLBACK_KEYS})

    def inspect(self,failed_key=None):
        def query(command,**kwargs):
            self.assertEqual(command[:3],['hermes','config','get'])
            self.assertEqual(command[3],'--json')
            self.assertIn(command[-1],checker.READ_KEYS)
            self.assertNotIn(command[-1],['fallback_model','fallback_providers','api_key','auth','providers'])
            return SimpleNamespace(returncode=int(command[-1]==failed_key),stdout=json.dumps(self.values[command[-1]]),stderr='')
        with patch.object(checker.subprocess,'run',side_effect=query):return checker.inspect('hermes',self.soul)

    def test_approved_local_profile_passes(self):
        self.assertEqual(self.inspect()['errors'],[])
        self.assertEqual(self.inspect()['warnings'],[])

    def test_preferences_warn_without_failing_policy(self):
        self.values.update({'model.default':'another-local-model','model.context_length':32768})
        self.soul.write_text('Existing user personality\n')
        result=self.inspect()
        self.assertFalse(result['errors']);self.assertEqual(len(result['warnings']),3)
        self.assertEqual(self.soul.read_text(),'Existing user personality\n')

    def test_remote_providers_endpoints_and_credential_urls_fail(self):
        for key,value in [('model.provider','openrouter'),('model.base_url','https://example.invalid/v1'),
                          ('model.base_url','http://localhost.evil.invalid:8080/v1'),
                          ('model.base_url','http://user:synthetic@127.0.0.1:8080/v1'),
                          ('auxiliary.vision.provider','auto'),('auxiliary.compression.base_url','https://example.invalid/v1')]:
            with self.subTest(key=key,value=value):
                values=dict(self.values);values[key]=value
                self.assertTrue(checker.evaluate(values)['errors'])

    def test_fallback_external_memory_and_unapproved_capabilities_fail(self):
        for key,value in [('security.allow_lazy_installs',True),('fallback_model.provider','cloud'),('fallback_providers.0.model','remote'),
                          ('memory.provider','honcho'),('memory.user_profile_enabled',True),('memory.write_approval',False),
                          ('platform_toolsets.cli',['terminal','web']),('plugins.enabled',['blender','messenger']),
                          ('skills.disabled',[])]:
            with self.subTest(key=key):
                values=dict(self.values);values[key]=value
                self.assertTrue(checker.evaluate(values)['errors'])

    def test_approved_attended_tools_are_not_rejected(self):
        self.values['platform_toolsets.cli']=['terminal','file','blender']
        result=self.inspect();self.assertFalse(result['errors']);self.assertTrue(result['warnings'])

    def test_unknown_security_field_fails_closed(self):
        self.assertTrue(self.inspect('memory.provider')['errors'])
        self.assertFalse(self.inspect('model.context_length')['errors'])

    def test_malformed_selected_values_fail_without_crashing(self):
        for key,value in [('model.base_url',42),('platform_toolsets.cli',[{}]),('skills.disabled',[{}])]:
            values=dict(self.values);values[key]=value
            self.assertTrue(checker.evaluate(values)['errors'])

    def test_explicit_unset_fallback_leaf_is_distinct_from_query_failure(self):
        def query(command,**kwargs):
            key=command[-1]
            if key in checker.FALLBACK_KEYS:return SimpleNamespace(returncode=1,stdout='',stderr='Config key not set: '+key)
            return SimpleNamespace(returncode=0,stdout=json.dumps(self.values[key]),stderr='')
        with patch.object(checker.subprocess,'run',side_effect=query):
            self.assertFalse(checker.inspect('hermes',self.soul)['errors'])

    def test_yaml_style_list_output_is_parsed(self):
        self.assertEqual(checker.parse('- terminal\n  - file'),['terminal','file'])

    def test_verbatim_quoted_provider_is_rejected_by_json_readback(self) -> None:
        """Do not decode display quotes as if they were actual config values."""
        self.values['model.provider']='"custom"'
        def query(command,**kwargs):
            value=self.values[command[-1]]
            # This is the official CLI distinction: text output echoes a
            # string verbatim, whereas --json preserves its actual type/value.
            output=json.dumps(value) if '--json' in command else str(value)
            return SimpleNamespace(returncode=0,stdout=output,stderr='')
        with patch.object(checker.subprocess,'run',side_effect=query):
            self.assertIn('Unapproved inference provider: model.provider',checker.inspect('hermes',self.soul)['errors'])

    def test_non_json_selected_value_is_unverified(self) -> None:
        """Malformed machine-readable output cannot establish policy compliance."""
        def query(command,**kwargs):
            key=command[-1]
            return SimpleNamespace(returncode=0,stdout='custom' if key=='model.provider' else json.dumps(self.values[key]),stderr='')
        with patch.object(checker.subprocess,'run',side_effect=query):
            self.assertIn('UNVERIFIED setting: model.provider',checker.inspect('hermes',self.soul)['errors'])


if __name__=='__main__':unittest.main()
