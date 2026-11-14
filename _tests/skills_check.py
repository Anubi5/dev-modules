"""Agent Skills format (agentskills.io) for every module skill, offline.

SKILL.md starts with YAML frontmatter; name is 1-64 lowercase letters, digits and single hyphens and equals the
folder; description is 1-1024 characters; only the specification's fields are used; the body stays under 500 lines.
"""
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


def main():
    """Check every skills/*/ folder of every module and the template."""
    bad = 0
    for skill in sorted(ROOT.glob('*/skills/*/')):
        if not (skill / 'SKILL.md').is_file():
            print(f'  {skill.relative_to(ROOT)}: no SKILL.md'); bad = 1; continue
        for problem in problems(skill):
            print(f'  {skill.relative_to(ROOT)}: {problem}'); bad = 1
    print('  skill format: OK' if not bad else '  skill format: FAILED')
    return bad


if __name__ == '__main__':
    sys.exit(main())
