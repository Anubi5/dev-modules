"""Offline compatibility checks for extracted generation capabilities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import gen
from generation import image, models


class GenerationTests(unittest.TestCase):
    def test_existing_workflow_graphs_are_identical(self):
        baseline = json.loads(Path(__file__).with_name("tests").joinpath("workflow-baseline.json").read_text())
        actual = [hashlib.sha256(json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                  for workflow in gen.all_workflows(include_packages=False)]
        self.assertEqual(actual, baseline["workflows"])

    def test_capabilities_import_without_cli_gpu_or_service_runtime(self):
        code = ("import sys; import generation.image,generation.video,generation.mesh; "
                "assert 'gen' not in sys.modules; assert 'torch' not in sys.modules; "
                "assert 'assets' not in sys.modules")
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parent), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", "-c", code], env=environment, check=True, timeout=10)

    def test_profile_override_does_not_change_the_catalogue(self):
        profile = models.profile()
        old = models.profile()["file"]
        profile["file"] = "caller-override.safetensors"
        self.assertEqual(models.profile()["file"], old)
        self.assertIs(gen.PROFILES, models.PROFILES)

    def test_image_adapter_rejects_a_different_architecture(self):
        with self.assertRaisesRegex(ValueError, "needs an SDXL model"):
            image.sdxl("prompt", "negative", 1, 20, 1024, 1024, False, "test",
                       prof={"arch": "another", "name": "fixture"})

    def test_recipe_rejects_an_unsupported_choice(self):
        recipe = {"workflow": {}, "inputs": {"texture": {"choices": [1024, 2048], "set": []}}}
        with self.assertRaisesRegex(SystemExit, "choose one of"):
            gen.apply_recipe(recipe, {"texture": 3000})

    def test_packages_separate_capabilities_and_copy_nested_settings(self):
        self.assertEqual(models.package(kind="video").architecture, "wan22")
        self.assertEqual(models.package(kind="3d").kind, "3d")
        first = models.package("realvis")
        first.settings["file"] = "changed"
        self.assertNotEqual(models.package("realvis").settings["file"], "changed")
        with self.assertRaisesRegex(ValueError, "capability mismatch"):
            models.package(gen.PROFILES["default_3d"], kind="image")

    def test_incomplete_video_package_fails_before_workflow_construction(self):
        from unittest.mock import patch
        with patch.dict(models.PROFILES["video_models"], {"broken": {"kind": "video", "arch": "wan22"}}):
            with self.assertRaisesRegex(ValueError, "Incomplete model package"):
                models.package("broken", kind="video")

    def test_video_model_package_is_data_and_cannot_select_python_code(self):
        from unittest.mock import patch
        from generation.video import wan
        profile = models.package(kind="video").settings
        profile["file"] = "local-alternative.gguf"
        with patch.dict(models.PROFILES["video_models"], {"alternative": profile}):
            workflow = wan("prompt", "negative", 1, 20, 33, "test", model="alternative")
            self.assertEqual(workflow["1"]["inputs"]["unet_name"], "local-alternative.gguf")
            profile["arch"] = "arbitrary_python_module"
            with self.assertRaisesRegex(ValueError, "needs architecture wan22"):
                wan("prompt", "negative", 1, 20, 33, "test", model="alternative")

    def test_video_cli_uses_the_selected_package_without_running_a_service(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        selected = models.package(kind="video").settings
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(models.PROFILES["video_models"], {"fixture": selected}), \
             patch.object(sys, "argv", ["gen.py", "video", "a scene", "--model", "fixture", "--seed", "7"]), \
             patch.object(gen, "OUTPUT", directory), patch.dict(gen.JOB, {}, clear=True), \
             patch.object(gen, "generate", return_value=(True, ["clip.webp"], None)) as execute, \
             patch.object(gen, "write_record", return_value=None) as record, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen.main(), 0)
            self.assertEqual(execute.call_args.args[0]["1"]["inputs"]["unet_name"], selected["file"])
            self.assertEqual(record.call_args.args[1]["model"], "fixture")


if __name__ == "__main__":
    unittest.main()
