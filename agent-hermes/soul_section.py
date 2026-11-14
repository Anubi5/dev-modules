"""Add or refresh the agent-hermes managed section of SOUL.md without touching the owner's own text.

The section is delimited by begin/end marker lines. A section written by older versions (no markers) is replaced only
when it matches the known unmarked text line for line (the model-store line may name any path); anything else is left
unchanged and reported, so personal text before or after the section can never be consumed.
"""
import sys
from pathlib import Path

LEGACY_VARIABLE_LINE = 4   # the model-store line, whose path wording changed between versions


def refresh(template: str, text: str) -> tuple[str, str]:
    """Return (new SOUL text, action): added, refreshed, migrated, unchanged or review."""
    if not template.endswith('\n'):
        template += '\n'
    lines = template.splitlines(keepends=True)
    begin, end, body = lines[0], lines[-1], lines[1:-1]
    if template in text:
        return text, 'unchanged'
    have = text.splitlines(keepends=True)
    if begin in have:
        start = have.index(begin)
        if have.count(begin) != 1 or end not in have[start + 1:]:
            return text, 'review'
        stop = have.index(end, start + 1)
        return ''.join(have[:start] + lines + have[stop + 1:]), 'refreshed'
    if body[0] in have:
        start = have.index(body[0]); old = have[start:start + len(body)]
        same = len(old) == len(body) and all(
            a == b or (i == LEGACY_VARIABLE_LINE and a.startswith('Weights are user data ')
                       and a.rstrip('\n').endswith('(/models, read only); never fetch,'))
            for i, (a, b) in enumerate(zip(old, body)))
        if not same:
            return text, 'review'
        return ''.join(have[:start] + lines + have[start + len(body):]), 'migrated'
    return text + ('\n' if text and not text.endswith('\n\n') else '') + template, 'added'


if __name__ == '__main__':
    template_path, soul_path = Path(sys.argv[1]), Path(sys.argv[2])
    current = soul_path.read_text() if soul_path.is_file() else ''
    updated, action = refresh(template_path.read_text(), current)
    if updated != current:
        soul_path.write_text(updated)
    if action == 'review':
        print('  CHECK SOUL.md has a changed EVA local profile section - left untouched; review it by hand')
    elif action != 'unchanged':
        print(f'  local profile section in SOUL.md {action}')
