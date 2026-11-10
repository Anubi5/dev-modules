"""Read-only native parser export; does not launch Hermes or call its main()."""
import importlib.util
import json
from pathlib import Path
import sys
import ast
import logging
import os

sys.dont_write_bytecode = True
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('help_export', str(Path(__file__).with_name('export_help.py')))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def guard(event, arguments):
    if event == 'open':
        mode, flags = arguments[1:3]
        if (mode and any(c in mode for c in 'wax+')) or flags & 0x643:
            raise RuntimeError('Parser export attempted a file write')
    if event == 'os.mkdir' and Path(arguments[0]).is_dir():
        return # mkdir(exist_ok=True) against an existing directory changes nothing.
    if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'subprocess.Popen', 'socket.connect'):
        raise RuntimeError('Parser export attempted '+event)


sys.addaudithook(guard)
from pm.environments import selected_venv, site_packages
sys.path.insert(0, str(site_packages(selected_venv(root))))
os.environ['HERMES_HOME'] = os.environ['HERMES_HELP_HOME']
# Compile only the native registration function and its builder imports. Command
# handlers are never executed; startup/repair code is deliberately not imported.
tree = ast.parse((root/'hermes_cli/main.py').read_text())
nodes = [node for node in tree.body if isinstance(node, ast.ImportFrom)
         and node.module and node.module.startswith('hermes_cli.subcommands.')]
builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_build_cli_parser')
namespace = dict(logger=logging.getLogger('help-export'))
for node in ast.walk(builder):
    if isinstance(node, ast.Name) and node.id.startswith('cmd_'):
        namespace[node.id] = lambda *args, **kwargs: None
namespace['_register_plugin_cli_commands'] = lambda subparsers: None
exec(compile(ast.Module(body=nodes+[builder],type_ignores=[]), '<native Hermes parser builders>', 'exec'), namespace)
parser, _ = namespace['_build_cli_parser']()
pages = {}


def walk(path, parser):
    record = helper.record(parser)
    record['usage'] = record['usage'].replace('hermes', 'ai hermes', 1)
    pages[path] = record
    import argparse
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                walk(path+' '+name, child)


walk('ai hermes', parser)
print(json.dumps(pages, indent=2))
