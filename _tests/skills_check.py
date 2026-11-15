"""Agent Skills format (agentskills.io) for every module skill, offline.

SKILL.md starts with YAML frontmatter; name is 1-64 lowercase letters, digits and single hyphens and equals the
folder; description is 1-1024 characters; only the specification's fields are used; the body stays under 500 lines.
Every module command a skill names in code ("ai gen image", "img-check") exists: "ai NAME SUB" must be an entry in
the effects of the module providing bin/ai-NAME (manifest v2), which _tests/manifest_contract_test.py checks against
that command's own help. Platform commands (ai mod, ai free, ...) are not module commands and are not checked here.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {'name', 'description', 'license', 'compatibility', 'metadata', 'allowed-tools'}
NAME = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


def frontmatter(text):
    """Top-level keys of the leading --- block (values unquoted); None when there is no block."""
    lines = text.split('\n')
    if not lines or lines[0].strip() != '---':
        return None, len(lines)
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == '---')
    except StopIteration:
        return None, len(lines)
    keys = {}
    for line in lines[1:end]:
        match = re.match(r'^([A-Za-z][\w-]*):\s*(.*)$', line)
        if match:
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            keys[match.group(1)] = value
    return keys, len(lines) - end - 1


def problems(skill):
    """Format problems of one skill folder."""
    text = (skill / 'SKILL.md').read_text(encoding='utf-8')
    keys, body = frontmatter(text)
    if keys is None:
        return ['no YAML frontmatter']
    found = []
    name, description = keys.get('name', ''), keys.get('description', '')
    if not (1 <= len(name) <= 64 and NAME.match(name)):
        found.append(f'name {name!r} is not 1-64 lowercase letters, digits and single hyphens')
    if name != skill.name:
        found.append(f'name {name!r} differs from the folder {skill.name!r}')
    if not 1 <= len(description) <= 1024:
        found.append(f'description has {len(description)} characters (1-1024)')
    if len(keys.get('compatibility', '')) > 500:
        found.append('compatibility longer than 500 characters')
    found += [f'unknown frontmatter field {k}' for k in keys if k not in FIELDS]
    if body >= 500:
        found.append(f'body has {body} lines (keep it under 500; move detail to references/)')
    return found


def commands():
    """Module command -> its effects keys (v2 manifests) or None (v1: the command exists, subcommands unknown)."""
    found = {}
    for module in sorted(p for p in ROOT.iterdir() if (p / 'bin').is_dir()):
        manifest = module / 'module.json'
        data = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
        keys = set(data.get('effects', {})) if data.get('schema') == 2 else None
        for command in (module / 'bin').iterdir():
            found[command.name] = None if keys is None else {k for k in keys if k.split(' ')[0] == command.name}
    return found


def named_commands(text):
    """(command, subcommand or None) for each "ai NAME [SUB]" inside inline code or fenced blocks."""
    code = re.findall(r'```.*?```', text, re.S) + re.findall(r'`[^`\n]+`', re.sub(r'```.*?```', '', text, flags=re.S))
    for span in code:
        for match in re.finditer(r'(?<![\w/.-])ai ([a-z][a-z0-9-]*)(?: ([a-z0-9][a-z0-9-]*))?', span):
            yield 'ai-' + match.group(1), match.group(2)


def unknown_commands(skill, known):
    """Module commands the skill names that the module does not declare."""
    found = []
    for command, sub in named_commands((skill / 'SKILL.md').read_text(encoding='utf-8')):
        keys = known.get(command)
        if not keys:
            continue                                       # platform command, or a v1 module (not yet declared)
        if sub and command not in keys and f'{command} {sub}' not in keys:
            found.append(f'names "{command[3:]} {sub}" but ai {command[3:]} has no such subcommand')
    return found


def main():
    """Check every skills/*/ folder of every module and the template."""
    bad = 0
    known = commands()
    for skill in sorted(ROOT.glob('*/skills/*/')):
        if not (skill / 'SKILL.md').is_file():
            print(f'  {skill.relative_to(ROOT)}: no SKILL.md'); bad = 1; continue
        for problem in problems(skill) + unknown_commands(skill, known):
            print(f'  {skill.relative_to(ROOT)}: {problem}'); bad = 1
    print('  skill format: OK' if not bad else '  skill format: FAILED')
    return bad


if __name__ == '__main__':
    sys.exit(main())
