import copy
import unittest
from validate_contract import load_contract, validate_summary


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()
        self.summary = dict(bones=list(self.contract["required_bones"]), root_parent=None,
                            pelvis_parent="root", root_deforms=False, max_influences=4,
                            unweighted_vertices=0, max_weight_sum_error=0, triangles=50000,
                            height_m=1.7, uv_layers=1, materials=1)

    def test_valid_rig(self):
        self.assertEqual(validate_summary(self.summary, self.contract), [])

    def test_each_mutation_is_rejected(self):
        for key, value in dict(bones=["root", "pelvis"], pelvis_parent=None,
                               root_deforms=True, max_influences=5, unweighted_vertices=1,
                               max_weight_sum_error=.1, triangles=100001, height_m=.85,
                               uv_layers=0, materials=0).items():
            with self.subTest(field=key):
                bad = copy.deepcopy(self.summary); bad[key] = value
                self.assertTrue(validate_summary(bad, self.contract))

    def test_duplicate_bones_rejected(self):
        self.summary["bones"].append("pelvis")
        self.assertIn("duplicate bone names", validate_summary(self.summary, self.contract))


if __name__ == "__main__":
    unittest.main()
