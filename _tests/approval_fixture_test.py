"""Exercise the synthetic approval backend offline; never counts as owner PC approval."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'_tests/prompt_fixture.py'


class FixtureTests(unittest.TestCase):
    def run_fixture(self,*args):
        result=subprocess.run([sys.executable,'-B',str(FIXTURE),*args],capture_output=True,text=True,timeout=10)
        return result,json.loads(result.stdout)

    def create(self):
        bundle=dict(fetch=(ROOT/'comfyui/fetch_models.py').read_text(),planner='')
        data=base64.b64encode(gzip.compress(json.dumps(bundle).encode())).decode()
        result,record=self.run_fixture('create',data)
        self.assertEqual(result.returncode,0,result.stderr)
        return record['fixture']

    def test_approved_synthetic_bytes_publish_and_unapproved_do_not(self):
        sha=hashlib.sha256(b'Owner-declined synthetic test file').hexdigest()
        root=self.create()
        result,record=self.run_fixture('apply',root,sha)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((record['published_sha256'],record['published_bytes'],record['synthetic_transfers'],record['staging_files']),(sha,34,1,0))
        root=self.create()
        result,record=self.run_fixture('apply',root)
        self.assertEqual(result.returncode,1)
        self.assertEqual((record['synthetic_transfers'],record['published_bytes']),(0,0))


if __name__=='__main__':unittest.main()
