"""Architecture rules that keep modules independent (offline; plan V3.2 section 3.7).

1. Every module has a manifest v2 (module.json with "schema": 2).
2. No module code reaches into another module: no imports or paths into another module's folder
   (~/.local/share/modules/<other>, ../<other>/...). Modules cooperate through commands and JSON contracts only.
   Test files may use other modules' sources as fixtures.
3. No module writes into another module or into Hermes: only agent-hermes runs "hermes" management commands or
   names ~/.hermes; other modules declare MCP servers and skills in module.json instead.
4. Every ComfyUI asset (folder/file) is declared once in comfyui/models.txt.

KNOWN lists today's violations with the plan step that removes them. An entry that no longer occurs fails too, so
the list only shrinks.
"""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
KNOWN = {
    ('comfyui', 'catalogue.py', 'rig'): 'Phase 4: read rig model declarations through a command/JSON contract',
    ('comfyui', 'catalogue.py', 'llama-swap'): 'Phase 4: read chat model declarations through a command/JSON contract',
    ('rig', 'cli.py', 'blender'): 'Phase 4: use the blender command instead of its private path',
    ('agent-hermes', 'hermes_eval.sh', 'comfyui'): 'Phase 4: use img-check and the skill instead of private paths',
}
TEXT = ('.py', '.sh', '.json', '.txt', '.service')


def modules():
    """Module folders (they have install.sh), excluding _template and _tests."""
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(('_', '.')) and (p / 'install.sh').is_file())


def is_test(path):
    """Test sources may reference other modules as fixtures."""
    name = path.name
    return 'tests' in path.parts or name.startswith('test_') or name.endswith(('_test.py', '_test.sh', '_pc.py'))


def code_files(module):
    """Module files that ship as code or configuration (bin/ scripts have no extension)."""
    for path in sorted(module.rglob('*')):
        if path.is_file() and (path.suffix in TEXT or path.parent.name == 'bin') and 'app' not in path.relative_to(module).parts:
            yield path


def found_violations():
    """Every (module, file, target) that breaks rules 2-4."""
    names = [m.name for m in modules()]
    found = set()
    for module in modules():
        others = '|'.join(re.escape(n) for n in names if n != module.name)
        reach = re.compile(rf"(modules/({others})\b|\.\./({others})/|['\"]({others})['\"]\s*/)")
        for path in code_files(module):
            if is_test(path):
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
            for match in reach.finditer(text):
                found.add((module.name, path.name, next(g for g in match.groups()[1:] if g)))
            if module.name != 'agent-hermes' and re.search(r'(^|[\s;(&|`])hermes (plugins|mcp|config|skills|tools)\b|\.hermes/', text, re.M):
                found.add((module.name, path.name, 'hermes'))
    seen = {}
    declarations = ROOT / 'comfyui/models.txt'
    for line in declarations.read_text(encoding='utf-8').splitlines() if declarations.is_file() else []:
        fields = line.split()
        if not fields or fields[0].startswith('#') or len(fields) < 3:
            continue
        saved = fields[3] if len(fields) > 3 and not re.match(r'^(sha256:|bytes:|optional$)', fields[3]) else fields[1].rsplit('/', 1)[-1]
        key = f'{fields[2]}/{saved}'
        if key in seen:
            found.add(('comfyui', 'models.txt', key))
        seen[key] = line
    return found


def main():
    """Report new violations and KNOWN entries that are already fixed."""
    bad = 0
    for module in modules():
        manifest = module / 'module.json'
        data = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
        if data.get('schema') != 2:
            print(f'  {module.name}: no manifest v2 (module.json with "schema": 2)'); bad = 1
    found = found_violations()
    for item in sorted(found - set(KNOWN)):
        print(f'  {item[0]}/{item[1]}: reaches into {item[2]} (cooperate through commands, JSON or module.json declarations)'); bad = 1
    for item in sorted(set(KNOWN) - found):
        print(f'  {item[0]}/{item[1]} -> {item[2]} is fixed: remove it from KNOWN in {Path(__file__).name}'); bad = 1
    print(f'  architecture: OK ({len(KNOWN)} known violations left)' if not bad else '  architecture: FAILED')
    return bad


if __name__ == '__main__':
    sys.exit(main())
