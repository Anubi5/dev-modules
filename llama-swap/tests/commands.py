"""llama-swap commands for the manifest effects contract (see _tests/manifest_contract_test.py)."""
from pathlib import Path
import re
import sys

MODULE = Path(__file__).resolve().parents[1]
CATALOGUE = MODULE.parent / 'comfyui' / 'catalogue.py'


def subcommands():
    """ai-models: the bare command (chat models) and the subcommands its help line lists."""
    script = (MODULE / 'bin/ai-models').read_text(encoding='utf-8')     # the help case echoes one literal usage line
    usage = re.search(r"help\|-h\|--help\) echo '([^']+)'", script).group(1)
    return {'ai-models': [''] + sorted(set(re.findall(r'(?:\[|\| )([a-z]+) ', usage)))}


def read_only_fixture(key, home):
    """Run the real script; a fake curl answers for the chat server, the comfyui catalogue answers the inventory."""
    tools = home / 'tools'
    tools.mkdir()
    (tools / 'curl').write_text('#!/bin/sh\necho \'{"data": [{"id": "qwen3.5-9b"}]}\'\n')
    (tools / 'ai-model-catalogue').write_text(f'#!/bin/sh\nexec "{sys.executable}" -B "{CATALOGUE}" "$@" --root "{home}/models"\n')
    for tool in tools.iterdir():
        tool.chmod(0o755)
    (home / 'models').mkdir()
    sub = key.split(' ', 1)[1:]
    argv = ['bash', str(MODULE / 'bin/ai-models')] + (sub + ['missing-id'] if sub in (['show'], ['verify']) else sub)
    return argv, {'PATH': f'{tools}:/usr/bin:/bin'}
