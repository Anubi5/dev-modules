"""Structured producer results do not depend on human progress messages."""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

import cli


class JsonTests(unittest.TestCase):
    def test_json_result_and_progress_have_separate_streams(self):
        def execute(args):
            self.assertTrue(args.json)
            print("arbitrary progress")
            cli.RESULT.update(status="succeeded", record="/work/record.json", outputs=["/work/model.fbx"])
            return 0
        with patch.object(sys, "argv", ["ai-rig", "model.glb", "--json"]), patch.object(cli, "execute", execute), \
             contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as progress:
            self.assertEqual(cli.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["schema"], "local-ai.producer/1")
        self.assertEqual(progress.getvalue().strip(), "arbitrary progress")

    def test_failure_and_parse_error_emit_json_with_original_exit_code(self):
        for args, code in ((["model.glb", "--json"], 1), (["--json"], 2)):
            with patch.object(sys, "argv", ["ai-rig"]+args), patch.object(cli, "execute", side_effect=RuntimeError("fixture failure")), \
                 contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(), code)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exit_code"], code)


if __name__ == "__main__": unittest.main()
