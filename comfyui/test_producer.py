"""Exercise the actual producer CLI/parser without services or GPU inference."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gen


class ProducerTests(unittest.TestCase):
    def test_real_cli_accepts_pipeline_options_before_literal_prompt(self):
        for kind in ("image", "video"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, \
                 patch.object(sys, "argv", ["gen.py", kind, "--json", "--name=test", "--seed=7", "--", "--literal scene"]), \
                 patch.dict(gen.JOB, {}, clear=True), patch.object(gen, "OUTPUT", directory), \
                 patch.object(gen, "generate", return_value=(True, [str(Path(directory)/"output.webp")], None)), \
                 contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as progress:
                self.assertEqual(gen.entry(), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["schema"], "local-ai.producer/1")
                self.assertEqual(result["exit_code"], 0)
                record = json.loads(Path(result["record"]).read_text())
                self.assertEqual(record["prompt"], "--literal scene")
                self.assertIn("OUTPUT", progress.getvalue())

    def test_json_preflight_failure_preserves_exit_code(self):
        with patch.object(sys, "argv", ["gen.py", "3d", "--json"]), \
             contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gen.entry(), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["record"])

    def test_json_parse_error_is_not_reported_as_a_success(self):
        with patch.object(sys, "argv", ["gen.py", "image", "a scene", "--json", "--unknown-option"]), \
             contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gen.entry(), 2)
        self.assertEqual(json.loads(output.getvalue())["exit_code"], 2)


if __name__ == "__main__": unittest.main()
