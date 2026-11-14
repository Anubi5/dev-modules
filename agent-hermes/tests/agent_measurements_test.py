"""SSE timings distinguish role events, generated content and missing usage."""
import io
import unittest
from unittest.mock import Mock

from agent_measurements import stream_chat, gpu_idle_gate


class StreamTests(unittest.TestCase):
    def test_default_idle_gate_is_unchanged(self):
        self.assertEqual(gpu_idle_gate(5)['limit_percent'],5)
        with self.assertRaises(RuntimeError):gpu_idle_gate(6)

    def test_explicit_baseline_is_recorded_and_additional_load_is_rejected(self):
        gate=gpu_idle_gate(21,21)
        self.assertEqual(gate['baseline_percent'],21)
        self.assertEqual(gate['limit_percent'],26)
        with self.assertRaises(RuntimeError):gpu_idle_gate(27,21)
        for value in [-1,96,True]:
            with self.assertRaises(ValueError):gpu_idle_gate(0,value)

    def test_role_events_do_not_count_as_first_token(self):
        response=io.BytesIO(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            b'data: {"model":"local","choices":[{"delta":{"content":"red"}}]}\n'
            b'data: {"choices":[{"delta":{"content":" square"}}]}\n'
            b'data: {"usage":{"completion_tokens":3}}\ndata: [DONE]\n')
        opener=Mock();opener.open.return_value=response
        clock=iter([10,12,14,15]).__next__
        result=stream_chat('http://127.0.0.1:8080','local','test',clock=clock,opener=opener)
        self.assertEqual(result['first_token_seconds'],2)
        self.assertEqual(result['text'],'red square')
        self.assertEqual(result['completion_tokens_per_generation_second'],1)
        self.assertEqual(result['served_model'],'local')

    def test_missing_usage_is_not_invented(self):
        opener=Mock();opener.open.return_value=io.BytesIO(b'data: {"choices":[{"delta":{"content":"yes"}}]}\ndata: [DONE]\n')
        result=stream_chat('http://127.0.0.1:8080','local','test',opener=opener)
        self.assertIsNone(result['completion_tokens'])
        self.assertIsNone(result['completion_tokens_per_generation_second'])

    def test_remote_or_credential_endpoint_is_rejected_before_io(self):
        for url in ['https://example.invalid','http://user:synthetic@127.0.0.1:8080']:
            with self.assertRaises(ValueError):stream_chat(url,'local','test')


if __name__=='__main__':unittest.main()
