"""ComfyUI commands for the manifest effects contract: subcommands read from each command's real --help, and
fixtures that run every read_only entry from this source tree in a disposable HOME (see _tests/manifest_contract_test.py)."""
from pathlib import Path
import re
import subprocess
import sys

MODULE = Path(__file__).resolve().parents[1]
# A safetensors file with an empty header: enough for show/list to read a real file.
TINY = (2).to_bytes(8, 'little') + b'{}'
# A 1x1 grey PNG for img-check.
PNG = bytes.fromhex('89504e470d0a1a0a0000000d4948445200000001000000010800000000'
                    '3a7e9b550000000a49444154789c636800000082008177cd72b60000000049454e44ae426082')


def help_text(script):
    """What the command prints for --help (no GPU, network or state; checked by _tests/command_help_test.py)."""
    return subprocess.run([sys.executable, '-B', str(MODULE / script), '--help'], capture_output=True, text=True,
                          check=True, cwd=MODULE).stdout


def subcommands():
    """ai gen: the generation kinds and the "other commands" its help lists; ai-model-catalogue: its command choices."""
    gen = help_text('gen.py')
    kinds = re.search(r'\{([a-z0-9,]+)\}', gen).group(1).split(',')
    other = re.findall(r'^  ([a-z0-9-]+) ', gen.split('other commands:', 1)[1], re.M)
    catalogue = re.search(r'\{([a-z0-9,]+)\}', help_text('catalogue.py')).group(1).split(',')
    return {'ai-gen': kinds + other, 'ai-model-catalogue': catalogue}


def read_only_fixture(key, home):
    """Run the module source with models under home (MODELS_MOUNT); inputs are created before the snapshot."""
    models = home / 'models'
    (models / 'comfy/loras').mkdir(parents=True)
    (models / 'comfy/loras/tiny.safetensors').write_bytes(TINY)
    env = {'MODELS_MOUNT': str(models)}
    gen = [sys.executable, '-B', str(MODULE / 'gen.py')]
    catalogue = [sys.executable, '-B', str(MODULE / 'catalogue.py')]
    if key == 'img-check':
        (home / 'grey.png').write_bytes(PNG)
        return [sys.executable, '-B', str(MODULE / 'img_check.py'), str(home / 'grey.png')], env
    argv = {
        'ai-gen jobs': gen + ['jobs'],
        'ai-gen list': gen + ['list'],
        'ai-gen show': gen + ['show', 'tiny'],
        'ai-gen names': gen + ['names', 'loras'],
        'ai-gen models': gen + ['models'],
        'ai-model-catalogue list': catalogue + ['list', '--root', str(models)],
        'ai-model-catalogue show': catalogue + ['show', 'tiny', '--root', str(models)],
        'ai-model-catalogue verify': catalogue + ['verify', 'tiny', '--root', str(models)],
    }[key]
    return argv, env
