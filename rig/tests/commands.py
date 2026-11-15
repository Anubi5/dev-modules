"""rig commands for the manifest effects contract (_tests/manifest_contract_test.py); its commands have no subcommands."""
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]


def read_only_fixture(key, home):
    """The read_only entry runs the real script: it prints the module's declarations and writes nothing."""
    return {'ai-rig-models': ['bash', str(MODULE / 'bin/ai-rig-models')]}[key]
