import ast
from pathlib import Path
import unittest

from render_content import render_metrics


class RenderTests(unittest.TestCase):
    def test_flat_colours_transparency_darkness_and_empty_fail(self):
        for colour in ((1, 0, 0, 1), (0, 0, 1, 1), (1, 1, 1, 1),
                       (.001, .001, .001, 1), (.2, .6, .4, 1), (1, 0, 0, 0)):
            with self.subTest(colour=colour): self.assertFalse(render_metrics([colour]*100)["ok"])
        self.assertFalse(render_metrics([])["ok"])

    def test_real_spatial_variation_passes_in_one_or_more_channels(self):
        for pixels in ([(v/99, 0, 0, 1) for v in range(100)],
                       [(v/99, v/99, v/99, 1) for v in range(100)]):
            self.assertTrue(render_metrics(pixels)["ok"])

    def test_coverage_and_nonfinite_evidence_fail(self):
        pixels = [(0, 0, 0, 0)]*99 + [(1, 0, 0, 1)]
        self.assertFalse(render_metrics(pixels)["ok"])
        for value in (float("nan"), float("inf"), -1):
            self.assertFalse(render_metrics([(value, 0, 0, 1)])["ok"])

    def test_blender_verifier_uses_the_content_check(self):
        tree = ast.parse(Path(__file__).with_name("verify.py").read_text())
        self.assertTrue(any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "render_metrics"
                            for n in ast.walk(tree)), "original flattened RGB check must not return")


if __name__ == "__main__":
    unittest.main()
