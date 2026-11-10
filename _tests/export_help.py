"""Export native parser help at development time; Windows needs only the JSON result.

Run with the standard-library Linux Python. Parsing is intercepted before command
execution. The Rig parser is read from its existing function without importing
its runtime or changing the module.
"""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Captured(BaseException):
    def __init__(self, parser):
        self.parser = parser


def capture(parser, *args, **kwargs):
    raise Captured(parser)


def record(parser):
    rows = []
    children = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            summaries = {item.dest:item.help for item in action._choices_actions}
            children = {name:dict(description=child.description or summaries.get(name,'') or name)
                        for name, child in action.choices.items()}
            continue
        rows.append(dict(names=action.option_strings or [action.dest],
                         display=parser._get_formatter()._format_action_invocation(action),
                         help=action.help if action.help != argparse.SUPPRESS else '',
                         default=None if action.default == argparse.SUPPRESS or re.search(r'key|token|password|secret|credential',action.dest,re.I) else str(action.default),
                         choices=list(action.choices) if action.choices is not None else [],
                         required=action.required, nargs=str(action.nargs)))
    return dict(usage=parser.format_usage().strip().removeprefix('usage: '),
                description=parser.description or '', arguments=rows, children=children)


def dispatch(script, arguments):
    old_path, old_argv = sys.path[:], sys.argv[:]
    try:
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *arguments]
        with patch.object(argparse.ArgumentParser, 'print_help', capture), \
             patch('subprocess.Popen', side_effect=AssertionError('Help started a process')), \
             patch('socket.socket', side_effect=AssertionError('Help accessed network')), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script), run_name='__main__')
            except Captured as result:
                if script.name=='catalogue.py': result.parser.prog='ai models'
                return record(result.parser)
        raise AssertionError('Command did not expose its parser: '+str(arguments))
    finally:
        sys.path[:], sys.argv[:] = old_path, old_argv


def export():
    pages = {}
    root=dispatch(ROOT/'comfyui/gen.py',['--help'])
    gen=next(row['choices'] for row in root['arguments'] if row['names']==['kind'])[:]
    tree=ast.parse((ROOT/'comfyui/gen.py').read_text())
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    # The real dispatcher declares its additional commands in argv comparisons.
    for node in ast.walk(main):
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Subscript) and ast.unparse(node.left).startswith('sys.argv['):
            for value in node.comparators:
                if isinstance(value,(ast.List,ast.Tuple)):
                    for item in value.elts:
                        if isinstance(item,ast.Constant) and isinstance(item.value,str) and item.value not in ('help','-h','--help') and item.value not in gen: gen.append(item.value)
    for command in ['', *gen]:
        path = 'ai gen'+(' '+command if command else '')
        pages[path] = dispatch(ROOT/'comfyui/gen.py', [command, '--help'] if command else ['--help'])
    models=dispatch(ROOT/'comfyui/catalogue.py',['--help'])
    commands=next(row['choices'] for row in models['arguments'] if row['names']==['command'])
    for command in ['', *commands]:
        path = 'ai models'+(' '+command if command else '')
        pages[path] = dispatch(ROOT/'comfyui/catalogue.py', [command, '--help'] if command else ['--help'])
    character = dispatch(ROOT/'character/cli.py', ['--help'])
    pages['ai character'] = character
    for name, child in character['children'].items():
        pages['ai character '+name] = dispatch(ROOT/'character/cli.py', [name, '--help'])
    # This function contains only argparse declarations; preserve Rig untouched.
    tree = ast.parse((ROOT/'rig/cli.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parser')
    namespace = dict(argparse=argparse, Path=Path)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<existing rig parser>', 'exec'), namespace)
    rig=namespace['parser'](); rig.prog='ai rig'
    pages['ai rig'] = record(rig)
    return pages


if __name__ == '__main__':
    pages=export()
    if '--check' in sys.argv:
        if pages!=json.loads((ROOT/'command-help.json').read_text()): raise SystemExit('Stale module help; run python3 -B _tests/export_help.py and refresh command-help.json')
        print('PASS: module help matches real parser dispatch')
    else:
        print(json.dumps(pages, indent=2))
