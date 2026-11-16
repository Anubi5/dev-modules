"""Subcommands and read-only fixtures for the manifest effects contract (tests/contract/manifest_contract_test.py)."""
from pathlib import Path
import re
import subprocess
import sys

MODULE = Path(__file__).resolve().parents[1]


def subcommands():
    """Read the subcommands from the command's own --help (never a second list)."""
    text = subprocess.run([sys.executable, '-B', str(MODULE / 'bin/ai-example'), '--help'], capture_output=True, text=True, check=True).stdout
    return {'ai-example': re.search(r'\{([a-z0-9,-]+)\}', text).group(1).split(',')}


def read_only_fixture(key, home):
    """argv (or (argv, extra environment)) that runs one read_only entry in the disposable home."""
    return {'ai-example hello': [sys.executable, '-B', str(MODULE / 'bin/ai-example'), 'hello']}[key]
