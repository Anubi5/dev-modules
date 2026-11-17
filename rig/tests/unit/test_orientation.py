import json
from pathlib import Path
import math
import unittest

from orientation import verify_facing, verify_named_sides


def rotated(vector, angle):
    x, y, z = vector
    return (x*math.cos(angle)-y*math.sin(angle), x*math.sin(angle)+y*math.cos(angle), z)


class OrientationTests(unittest.TestCase):
    feet = ((.01, -.2, -.01), (-.01, -.2, -.02))
    hips = ((-.12, 0, 1), (.16, 0, 1.02))  # Asymmetric and intentionally unordered by name.

    def test_rotated_asymmetric_reference_normalizes_without_swapping_sides(self):
        for degrees, facing in ((0, "-y"), (90, "+x"), (180, "+y"), (270, "-x")):
            angle = math.radians(degrees)
            feet = [rotated(v, angle) for v in self.feet]
            hips = [rotated(v, angle) for v in self.hips]
            result = verify_facing(feet, hips, [1, 1], facing)
            normalized = [rotated(v, result["yaw_radians"]) for v in hips]
            self.assertLess(normalized[0][0], 0)  # Character right when facing -Y.
            self.assertGreater(normalized[1][0], 0)
            for actual, expected in zip(normalized, self.hips):
                for a, b in zip(actual, expected): self.assertAlmostEqual(a, b)

    def test_fixed_default_cannot_silently_accept_wrong_rotation(self):
        for degrees in (90, 180, 270):
            angle = math.radians(degrees)
            with self.assertRaisesRegex(ValueError, "Declared facing"):
                verify_facing([rotated(v, angle) for v in self.feet],
                              [rotated(v, angle) for v in self.hips], [1])

    def test_mirror_and_ambiguous_evidence_are_rejected(self):
        for determinants in ([-1], [0], [float("nan")]):
            with self.assertRaises(ValueError): verify_facing(self.feet, self.hips, determinants)
        for feet in ([], [(0, 0, 1)]*2, [self.feet[0], (0, .2, 0)], [(float("nan"), 1, 0)]*2):
            with self.assertRaises(ValueError): verify_facing(feet, self.hips, [1])
        with self.assertRaisesRegex(ValueError, "left/right"):
            verify_facing(self.feet, [(0, 0, 1), (0, 1, 1)], [1])

    def test_retained_unirig_coordinates_and_wrong_facing(self):
        fixtures = json.loads((Path(__file__).parent / "fixtures/unirig_feet.json").read_text())
        for fixture in fixtures:
            with self.subTest(fixture=fixture["file"]):
                chains = fixture["chains"]
                segments = [tuple(b-a for a, b in zip(c[2]["head"], c[3]["head"])) for c in chains]
                hips = [c[0]["head"] for c in chains]
                result = verify_facing(segments, hips, fixture["determinants"])
                self.assertEqual(result["reference"], "unirig-toe-v1")
                self.assertGreater(min(result["facing_alignments"]), .96)
                with self.assertRaisesRegex(ValueError, "Declared facing"):
                    verify_facing(segments, hips, fixture["determinants"], "+y")

    def test_reimport_rejects_structurally_valid_swapped_names(self):
        points = dict(pelvis=(0, 0, 1), hand_l=(1, 0, 1), hand_r=(-1, 0, 1),
                      thigh_l=(.16, 0, 1), thigh_r=(-.12, 0, 1))
        verify_named_sides(points)
        points["hand_l"], points["hand_r"] = points["hand_r"], points["hand_l"]
        with self.assertRaisesRegex(ValueError, "Semantic left/right"):
            verify_named_sides(points)


if __name__ == "__main__":
    unittest.main()
