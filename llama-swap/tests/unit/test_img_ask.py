import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import img_ask


def response(text, reason='stop'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': reason}]}


class AnswerTests(unittest.TestCase):
    def test_retained_real_grounded_captions_become_plain_text(self):
        for caption, box in (
            ('The character is a girl wearing a teal jacket and gray pants standing straight up with her arms at her sides.', '<229><95><796><946>'),
            ('The character is waving their arms', '<244><128><728><935>'),
        ):
            self.assertEqual(img_ask.answer_text(response('<ref>'+caption+'</ref><box>'+box+'</box>'), 'Describe the character.'), caption)

    def test_empty_echo_repetition_partial_and_malformed_outputs_fail(self):
        question = 'Is the character waving?'
        for text in ('', '   ', question, question+' Answer briefly.', 'Yes Yes Yes Yes',
                     'The arm is raised. The arm is raised. The arm is raised.',
                     '<ref>Waving</ref><box><1>', '<think>reasoning</think>', 'x'*601):
            with self.subTest(text=text), self.assertRaises(img_ask.InvalidAnswer):
                img_ask.answer_text(response(text), question)
        with self.assertRaises(img_ask.InvalidAnswer):
            img_ask.answer_text(response('The character is waving.', 'length'), question)
        for data in ({}, {'choices': []}, {'choices': [None]}, response(None), response(['yes'])):
            with self.assertRaises(img_ask.InvalidAnswer): img_ask.answer_text(data, question)

    def test_plain_concise_answer_is_preserved(self):
        self.assertEqual(img_ask.answer_text(response('No, both arms are lowered.'), 'Is one arm raised?'),
                         'No, both arms are lowered.')

    def test_request_stays_on_existing_model_and_bounds_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder)/'image.png'; image.write_bytes(b'image fixture')
            with patch.object(img_ask.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(response('A person waves.')).encode())) as request:
                self.assertEqual(img_ask.ask(str(image), 'Describe it.'), 'A person waves.')
            body=json.loads(request.call_args.args[0].data)
            self.assertEqual(body['model'], 'vision-check')
            self.assertEqual(body['max_tokens'], 128)
            self.assertEqual(body['temperature'], 0)

    def test_invalid_answer_cannot_satisfy_expect(self):
        output=io.StringIO()
        with patch.object(img_ask, 'check_image', return_value=('image.png',None)), \
                patch.object(img_ask, 'ask', side_effect=img_ask.InvalidAnswer('repeated tokens')), contextlib.redirect_stdout(output):
            self.assertEqual(img_ask.main(['image.png', 'Is it correct?', '--expect', 'yes']), 1)
        self.assertIn('NOT verified', output.getvalue())
        self.assertNotIn(' -> ', output.getvalue())

    def test_stopped_service_has_explicit_existing_start_command(self):
        output=io.StringIO()
        with patch.object(img_ask, 'check_image', return_value=('image.png',None)), \
                patch.object(img_ask, 'ask', side_effect=URLError(ConnectionRefusedError(111,'refused'))), contextlib.redirect_stdout(output):
            self.assertEqual(img_ask.main(['image.png', 'Describe it.']), 1)
        self.assertIn('llama-swap is not listening', output.getvalue())
        self.assertIn('ai start llama-swap', output.getvalue())


if __name__ == '__main__':
    unittest.main()
