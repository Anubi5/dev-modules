"""Deliberately reintroduce original decisions in memory; regressions must fail."""
import statistics
import unittest
from unittest.mock import patch

import test_chat_service
import test_orientation
import test_render_content
import test_source_integrity


class MutationTests(unittest.TestCase):
    def rejects(self, case):
        result = unittest.TestResult()
        case.run(result)
        self.assertTrue(result.failures, "Regression did not detect the original defect")
        self.assertFalse(result.errors, "A fixture crash is not evidence of detecting the defect")

    def test_original_flattened_rgb_decision_is_detected(self):
        def original(pixels):
            values = [v for row in pixels for v in row[:3]]
            return {"ok": bool(values) and statistics.pstdev(values) > .005}
        with patch.object(test_render_content, "render_metrics", original):
            self.rejects(test_render_content.RenderTests("test_flat_colours_transparency_darkness_and_empty_fail"))

    def test_original_unverified_facing_is_detected(self):
        with patch.object(test_orientation, "verify_facing", return_value={"yaw_radians": 0}):
            self.rejects(test_orientation.OrientationTests("test_fixed_default_cannot_silently_accept_wrong_rotation"))

    def test_original_chat_not_stopped_is_detected(self):
        with patch.object(test_chat_service.chat_service, "stop_chat", lambda: None):
            self.rejects(test_chat_service.ChatTests("test_stop_is_verified_and_bounded"))

    def test_original_tracked_only_source_check_is_detected(self):
        def original(source, revision):
            # This really succeeds with the harmless ignored/untracked fixture files.
            test_source_integrity.subprocess.check_output(["git", "-C", str(source), "diff", "--quiet"])
            return "0"*64
        with patch.object(test_source_integrity, "verify_source", original):
            self.rejects(test_source_integrity.SourceTests("test_untracked_and_ignored_executables_fail_even_with_clean_git_diff"))


if __name__ == "__main__":
    unittest.main()
