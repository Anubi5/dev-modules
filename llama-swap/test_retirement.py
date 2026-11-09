"""Retired declarations stay out of serving/benchmarks without touching weights."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import bench
import make_config

RETIRED = {'lfm2.5-24b', 'nemotron-bolt'}
SPEC = Path(__file__).with_name('models.json')


class RetirementTests(unittest.TestCase):
    """Include stale server/file records to exercise the actual retirement paths."""

    def test_stale_file_records_cannot_reenable_retired_profiles(self) -> None:
        """Generate only declared profiles and preserve both retained files."""
        cfg = json.loads(SPEC.read_text())
        self.assertFalse(RETIRED & {row['id'] for row in cfg['chat']})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            files = {}
            for name in RETIRED | {row['id'] for row in cfg['chat']}:
                weight = root / (name + '.gguf')
                weight.write_bytes(b'protected synthetic weight')
                records[name] = {'file': str(weight)}
                files[weight] = (weight.read_bytes(), weight.stat().st_mtime_ns)
            original = copy.deepcopy(records)
            with patch.object(make_config, 'AI', str(root)):
                config = make_config.build(cfg, records)
            for name in RETIRED:
                self.assertNotIn('"' + name + '":', config)
            self.assertEqual(records, original)
            self.assertEqual(files, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})

    def test_default_benchmark_excludes_retired_ids_even_on_stale_server(self) -> None:
        """A server still advertising an old profile cannot add it to the bench."""
        cfg = json.loads(SPEC.read_text())
        active = {row['id'] for row in cfg['chat']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models.json').write_text(json.dumps(cfg))
            advertised = {'data': [{'id': name} for name in sorted(active | RETIRED)]}
            with patch.object(bench, 'MOD', str(root)), patch.object(bench, 'OUT', str(root / 'bench.md')), \
                 patch.object(bench.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(advertised).encode())), \
                 patch.object(bench, 'run', return_value=['header', 'separator', '| fixture | yes |']) as run, \
                 patch.object(sys, 'argv', ['bench.py']):
                with self.assertRaises(SystemExit) as exit_result:
                    bench.main()
            self.assertEqual(exit_result.exception.code, 0)
            selected = set(run.call_args.args[0])
            self.assertEqual(selected, active)
            self.assertFalse(RETIRED & selected)


if __name__ == '__main__':
    unittest.main()
