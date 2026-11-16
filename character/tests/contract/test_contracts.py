import copy
from pathlib import Path
import tempfile
import unittest

import contracts


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root/"model.glb"; self.file.write_bytes(b"fixture, not geometry acceptance")
        self.path = self.root/"mesh.asset.json"
        self.data = contracts.write(self.path, "mesh", [("model", self.file)],
                                    dict(units="m", up="+Z", forward="-Y"))

    def test_hashes_and_geometry_declarations_round_trip(self):
        self.assertEqual(contracts.read(self.path), self.data)
        self.assertEqual(self.data["validation"]["numerical"], "not_run")

    def test_changed_file_is_rejected(self):
        self.file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            contracts.read(self.path)

    def test_portable_paths_and_symlink_escape_are_rejected(self):
        for path in ("../model.glb", "/model.glb", "C:\\model.glb"):
            with self.subTest(path=path), self.assertRaises(ValueError): contracts.inside(self.root, path)
        link = self.root/"outside"; link.symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError): contracts.inside(self.root, "outside/model.glb")

    def test_unknown_version_and_malformed_fields_fail_cleanly(self):
        for changes in ({"schema": "local-ai.asset/2"}, {"validation": []}, {"files": []}):
            data = dict(self.data, **changes)
            with self.assertRaises(ValueError): contracts.validate(data, self.root)
        data = copy.deepcopy(self.data); data["files"][0]["sha256"] = 7
        with self.assertRaises(ValueError): contracts.validate(data, self.root)

    def test_rig_is_experimental_and_manifest_is_not_overwritten(self):
        rig_file = self.root/'rig.fbx'; rig_file.write_bytes(b'fixture')
        data = contracts.write(self.root/"rig.asset.json", "rig", [("model", rig_file)],
                               dict(units="cm", up="+Z", forward="+X"), numerical="passed")
        self.assertTrue(data["experimental"])
        self.assertEqual(data["validation"]["visual"], "not_established")
        with self.assertRaises(ValueError): contracts.write(self.path, "image", [("model", self.file)])
        with self.assertRaisesRegex(ValueError, 'FBX-only'):
            contracts.write(self.root/'wrong-rig.asset.json', 'rig', [('model', self.file)],
                            dict(units='cm', up='+Z', forward='+X'))

    def test_retained_producer_record_is_part_of_integrity_check(self):
        record = self.root/"record.json"; record.write_text("{}")
        data = dict(self.data, source=dict(record=record.name, sha256=contracts.digest(record)))
        contracts.validate(data, self.root)
        record.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "producer record"): contracts.validate(data, self.root)
