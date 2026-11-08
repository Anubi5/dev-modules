"""Real hard-link and failure behavior on disposable native filesystem fixtures; no network."""
from pathlib import Path
import tempfile
import unittest

from publication_fixture import exercise,CASES


class PublicationTests(unittest.TestCase):
    def test_actual_publisher_preserves_existing_files_and_failed_staging(self):
        source=Path(__file__).resolve().parents[1].joinpath('comfyui/fetch_models.py').read_text()
        with tempfile.TemporaryDirectory() as directory:
            result=exercise(Path(directory),source)
            self.assertEqual(result['status'],'PASS',result)
            self.assertEqual(set(result['cases']),set(CASES))
            self.assertTrue(all(row['status']=='PASS' for row in result['cases'].values()))


if __name__=='__main__':unittest.main()
