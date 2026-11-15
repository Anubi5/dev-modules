"""character commands for the manifest effects contract (see _tests/manifest_contract_test.py)."""
from pathlib import Path
import re
import subprocess
import sys

MODULE = Path(__file__).resolve().parents[1]


def subcommands():
    """ai character: the subcommand choices its --help lists."""
    text = subprocess.run([sys.executable, '-B', str(MODULE / 'cli.py'), '--help'], capture_output=True, text=True, check=True).stdout
    return {'ai-character': re.search(r'\{([a-z,]+)\}', text).group(1).split(',')}


def read_only_fixture(key, home):
    """inspect a manifest and plan a pipeline that both refuse (reported, exit 1): the read path runs to the end."""
    (home / 'manifest.json').write_text('{}')
    (home / 'pipeline.json').write_text('{"stages": []}')
    target = {'ai-character inspect': ['inspect', str(home / 'manifest.json')],
              'ai-character plan': ['plan', str(home / 'pipeline.json')]}[key]
    return [sys.executable, '-B', str(MODULE / 'cli.py')] + target
