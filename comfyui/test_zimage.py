"""Native Z-Image controls and bounded real-parser batch regressions, without GPU services."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gen
from generation import image, models


class ZImageTests(unittest.TestCase):
    def test_native_pose_and_face_repair_keep_correct_architecture_inputs(self):
        p = models.profile("zimage")
        wf = image.zimage("adult portrait", 7, 8, 1024, 1024, "test", p,
                          pose="pose.png", fix_faces=True, hires=True, upscale=True)
        self.assertEqual(wf["33"]["class_type"], "QwenImageDiffsynthControlnet")
        self.assertEqual(wf["8"]["inputs"]["model"], ["33", 0])
        self.assertEqual(wf["41"]["inputs"]["clip"], ["2", 0])
        self.assertEqual(wf["41"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(wf["41"]["inputs"]["model"], ["7", 0])
        self.assertEqual(wf["41"]["inputs"]["cfg"], 1)
        self.assertEqual(wf["21"]["inputs"]["image"], ["41", 0])
        self.assertEqual(wf["12"]["inputs"]["image"], ["41", 0])
        self.assertEqual(wf["41"]["inputs"]["guide_size"], 1024)
        self.assertEqual(wf["41"]["inputs"]["denoise"], 0.15)
        self.assertEqual(wf["36"]["inputs"]["width"], 1024)
        self.assertEqual(wf["36"]["inputs"]["crop"], "disabled")
        gen.use_skeleton(wf)
        self.assertNotIn("31", wf)
        self.assertEqual(wf["33"]["inputs"]["image"], ["36", 0])
        self.assertEqual(wf["36"]["inputs"]["image"], ["30", 0])

    def batch(self, count, outcomes):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def generate(wf, *args, **kwargs):
                index = len(calls)
                calls.append(wf)
                return outcomes[index], [str(Path(directory)/f"output-{index}.png")], None if outcomes[index] else "fixture failure"
            with patch.object(sys, "argv", ["gen.py", "image", "adult portrait", "--model", "zimage", "--count", str(count), "--seed", "0", "--json"]), \
                 patch.object(gen, "OUTPUT", directory), patch.dict(gen.JOB, {}, clear=True), \
                 patch.object(gen, "resolve_model", return_value=models.profile("zimage")), \
                 patch.object(gen, "input_hashes", return_value={}), patch.object(gen, "generate", side_effect=generate), \
                 contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                code = gen.entry()
                result = json.loads(output.getvalue())
                records = [json.loads(Path(p).read_text()) for p in result.get("records", [])]
                return code, result, records, calls

    def test_batches_record_incrementing_seeds_and_distinct_names(self):
        code, result, records, calls = self.batch(3, [True]*3)
        self.assertEqual(code, 0)
        self.assertEqual([r["seed"] for r in records], [0, 1, 2])
        self.assertEqual([w["8"]["inputs"]["seed"] for w in calls], [0, 1, 2])
        self.assertEqual(len({w["10"]["inputs"]["filename_prefix"] for w in calls}), 3)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(len(result["outputs"]), 3)

    def test_batch_stops_at_first_failure_and_keeps_prior_records(self):
        code, result, records, calls = self.batch(3, [True, False])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([r["status"] for r in records], ["succeeded", "failed"])

    def test_invalid_batch_count_does_not_run(self):
        for count in (0, 9):
            code, result, records, calls = self.batch(count, [])
            self.assertEqual(code, 2)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
