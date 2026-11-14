"""Validate Hermes evidence parsing without inference, files, services or credentials."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hermes_batch2_pc as acceptance


class AcceptanceTests(unittest.TestCase):
    def events(self,tool=True,exit_code=0):
        events=[dict(type='system',subtype='init',model='qwen3.5-9b-agent')]
        if tool:events += [dict(type='tool_use',name='memory'),dict(type='tool_result',name='memory',is_error=False)]
        events.append(dict(type='result',exit_code=exit_code,text='synthetic'))
        return '\n'.join(map(json.dumps,events))

    def test_real_tool_evidence_is_required(self):
        self.assertEqual(acceptance.events_result(self.events(),tool='memory'),'synthetic')
        with self.assertRaises(ValueError):acceptance.events_result(self.events(False),tool='memory')
        with self.assertRaises(ValueError):acceptance.events_result(self.events(exit_code=1))

    def test_wrong_model_and_human_output_are_not_accepted(self):
        with self.assertRaises(ValueError):acceptance.events_result(self.events().replace('qwen3.5-9b-agent','unexpected'))
        with self.assertRaises(ValueError):acceptance.events_result('The agent says SUCCESS')

    def test_private_fixture_config_has_no_external_provider_or_fallback(self):
        config=acceptance.nested_settings()
        self.assertEqual(config['memory']['provider'],'none')
        self.assertTrue(config['memory']['memory_enabled'])
        self.assertFalse(config['memory']['user_profile_enabled'])
        self.assertEqual(config['fallback_providers'],[])

    def test_environment_projection_does_not_copy_credential_variables(self):
        with patch.dict(os.environ,UNRELATED_TEST_CREDENTIAL='synthetic-value'):
            env=acceptance.hermes_env('fixture')
        self.assertNotIn('UNRELATED_TEST_CREDENTIAL',env)
        self.assertEqual(env['OPENAI_API_KEY'],'local') # Explicit dummy for local OpenAI-compatible clients.
        self.assertEqual(env['HERMES_DISABLE_LAZY_INSTALLS'],'1')

    def test_live_harness_uses_production_ai_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            def spawn(command,**kwargs):
                self.assertEqual(command[:3],['ai','hermes','chat'])
                self.assertEqual(kwargs['env']['HERMES_DISABLE_LAZY_INSTALLS'],'1')
                kwargs['stdout'].write(self.events(False))
                return SimpleNamespace(wait=lambda **kw:0,poll=lambda:0)
            with patch.object(acceptance.subprocess,'Popen',side_effect=spawn):
                self.assertEqual(acceptance.run_agent('hermes',work,work,'synthetic','fixture'),'synthetic')


if __name__=='__main__':unittest.main()
