"""Exercise smoke protocol and diagnostics using only synthetic child processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SmokeTests(unittest.TestCase):
    """A prepared command is used directly and failed initialization stops early."""

    def run_fixture(self, fail: bool) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        """Run a fake prepared command and return smoke output and its exact argv."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / 'prepared'
            arguments = root / 'arguments.json'
            command.write_text('#!' + sys.executable + '\n' + '''
import json,sys
from pathlib import Path
Path(sys.argv[0]).with_name('arguments.json').write_text(json.dumps(sys.argv[1:]))
for line in sys.stdin:
 request=json.loads(line)
 if 'id' not in request:continue
 if FAIL:
  print('synthetic prepared-runtime failure',file=sys.stderr,flush=True)
  break
 method=request['method']
 if method=='initialize':result={}
 elif method=='tools/list':result={'tools':[{'name':n} for n in ['get_screenshot_of_window_as_image']+[str(i) for i in range(11)]]}
 else:result={'content':[{'type':'image','data':'synthetic'}]}
 print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}),flush=True)
'''.replace('FAIL', repr(fail)))
            command.chmod(0o700)
            # Any accidental return to uv must fail, even on a machine with uv.
            uv = root / 'uv'
            uv.write_text('#!/bin/sh\necho unexpected-resolver >&2\nexit 99\n')
            uv.chmod(0o700)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get('PATH', ''))
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('mcp_smoke.py')), str(command)], env=env, capture_output=True, text=True, timeout=15)
            return result, json.loads(arguments.read_text())

    def test_prepared_command_and_protocol(self) -> None:
        """Complete the synthetic scene/screenshot sequence without a resolver."""
        result, arguments = self.run_fixture(False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments, ['--transport', 'stdio'])
        self.assertIn('scene read (get_objects_summary): OK', result.stdout)
        self.assertIn('screenshot (get_screenshot_of_window_as_image): OK', result.stdout)
        self.assertNotIn('unexpected-resolver', result.stderr)

    def test_failed_initialize_keeps_child_diagnostic(self) -> None:
        """An exited server reports its cause without another protocol write."""
        result, _ = self.run_fixture(True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('MCP initialize: FAILED', result.stdout)
        self.assertIn('synthetic prepared-runtime failure', result.stderr)
        self.assertNotIn('BrokenPipeError', result.stderr)


if __name__ == '__main__':
    unittest.main()
