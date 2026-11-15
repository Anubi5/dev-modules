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
import runtime


class ProducerTests(unittest.TestCase):
    def test_real_cli_accepts_pipeline_options_before_literal_prompt(self):
        for kind in ("image", "video"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, \
                 patch.object(sys, "argv", ["gen.py", kind, "--json", "--name=test", "--seed=7", "--", "--literal scene"]), \
                 patch.dict(gen.JOB, {}, clear=True), patch.object(runtime, "OUTPUT", directory), \
                 patch.object(runtime, "generate", return_value=(True, [str(Path(directory)/"output.webp")], None)), \
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


class CheckTests(unittest.TestCase):
    """--check validates and reports, before any GPU lock, service or output (Plan 3.5)."""

    def check(self, argv, models, home):
        stop = AssertionError("--check reached the GPU / a service")
        with patch.object(sys, "argv", ["gen.py", *argv, "--check", "--json"]), patch.dict(gen.JOB, {}, clear=True), \
             patch.object(runtime, "MODELS", str(models)), patch.object(runtime.assets, "ROOTS", [str(models / "comfy")]), \
             patch.object(runtime, "OUTPUT", str(Path(home) / "out")), patch.object(runtime, "generate", side_effect=stop), \
             patch.object(runtime, "gpu_lease", side_effect=stop), patch.object(runtime, "expand_prompt", side_effect=stop), \
             patch.object(runtime, "model_server_up", side_effect=stop), patch.dict("os.environ", HOME=str(home)), \
             contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as progress:
            code = gen.entry()
        self.assertFalse((Path(home) / "out").exists(), "--check wrote output")
        return code, json.loads(output.getvalue()), progress.getvalue()

    def test_ready_job_reports_model_family_features_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models"
            for part in runtime.builders.registry.family(runtime.profile("zimage"))["parts"].values():
                if not part.get("optional"):
                    (models / "comfy" / part["folder"]).mkdir(parents=True, exist_ok=True)
                    (models / "comfy" / part["folder"] / part["file"]).write_bytes(b"x")
            code, result, progress = self.check(["image", "a mug", "--model", "zimage", "--seed", "5", "--upscale"], models, directory)
        self.assertEqual((code, result["status"], result["exit_code"]), (0, "checked", 0))
        self.assertEqual((result["model"], result["family"], result["features"], result["seed"]), ("zimage", "zimage", ["upscale"], 5))
        self.assertRegex(result["workflow_sha256"], "^[0-9a-f]{64}$")
        self.assertIn("CHECK OK", progress)

    def test_missing_files_are_named_with_the_approved_way_to_get_them(self):
        with tempfile.TemporaryDirectory() as directory:
            code, result, _ = self.check(["image", "a mug", "--model", "zimage"], Path(directory) / "models", directory)
        self.assertEqual((code, result["status"]), (1, "check_failed"))
        self.assertIn("diffusion_models/z_image_turbo_int8_convrot.safetensors", result["error"])
        self.assertIn("two-phase download plan", result["error"])

    def test_needed_chat_model_is_reported_not_started(self):
        with tempfile.TemporaryDirectory() as directory:
            code, result, _ = self.check(["image", "a sign", "--model", "flux3"], Path(directory) / "models", directory)
        self.assertEqual(code, 1)
        self.assertIn("llm.chat (module llama-swap)", result["error"])
        self.assertIn("--no-expand", result["error"])


if __name__ == "__main__": unittest.main()
