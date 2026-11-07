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
    def test_existing_233_workflow_graphs_are_identical(self):
        baseline = json.loads(Path(__file__).with_name("tests").joinpath("workflow-baseline.json").read_text())
        actual = [hashlib.sha256(json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                  for workflow in gen.all_workflows()]
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


if __name__ == "__main__":
    unittest.main()
