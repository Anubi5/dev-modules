"""Prepare Blender's pinned MCP command at install time; never install on launch."""
import argparse
import email.parser
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pin(source):
    match = re.fullmatch(r'git\+(https://projects\.blender\.org/lab/blender_mcp\.git)@([a-f0-9]{40})#subdirectory=mcp', source)
    if not match:
        raise ValueError('Blender MCP source must declare the reviewed full Git commit')
    return match.groups()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
    os.replace(name, path)


def prepare(module, executable, source, plugin=None):
    url, revision = source_pin(source)
    executable = executable.resolve(strict=True)
    records = list(executable.parent.parent.glob('lib/python*/site-packages/blender_mcp-*.dist-info'))
    if len(records) != 1 or not os.access(executable, os.X_OK):
        raise ValueError('Prepared Blender MCP environment is missing or ambiguous')
    provenance = records[0] / 'direct_url.json'
    direct = json.loads(provenance.read_text())
    if direct.get('url') != url or direct.get('vcs_info', {}).get('commit_id') != revision:
        raise ValueError('Prepared Blender MCP source does not match the declared pin')
    metadata = sorted(records[0].parent.glob('*.dist-info/METADATA'))
    if not metadata:
        raise ValueError('Prepared dependency metadata is missing')
    lock = module / 'mcp-requirements.lock'
    expected = {}
    normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    for line in lock.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+)', line)
        if not match or normalize(match[1]) in expected:
            raise ValueError('MCP dependency lock must contain unique exact versions')
        expected[normalize(match[1])] = match[2]
    actual = {}
    for path in metadata:
        fields = email.parser.Parser().parsestr(path.read_text())
        actual[normalize(fields['Name'])] = fields['Version']
    actual.pop('blender-mcp')  # server identity comes from its exact Git provenance
    if actual != expected:
        raise ValueError('Prepared Blender MCP dependencies differ from the declared lock')
    receipt = dict(source=source, executable=str(executable), executable_sha256=digest(executable),
                   requirements_sha256=digest(lock),
                   provenance=str(provenance), provenance_sha256=digest(provenance),
                   dependencies=[dict(path=str(path), sha256=digest(path)) for path in metadata])
    config = None
    if plugin is not None and (plugin / 'mcp.json').is_file():
        path = plugin / 'mcp.json'
        original = path.read_bytes()
        data = json.loads(original)
        servers = data.get('mcpServers', data.get('mcp_servers', data))
        server = servers.get('blender')
        # Agent Plugins v1 rejects absolute command paths. The module runner
        # already links this owned wrapper into ~/.local/bin on Hermes's PATH.
        command = 'blender-mcp-prepared'
        if not isinstance(server, dict):
            raise ValueError('Blender plugin server declaration is missing')
        arguments = server.get('args', [])
        if server.get('command') not in (command, str(module / 'bin/blender-mcp-prepared')):
            expected = ['tool', 'run', '--from', source, 'blender-mcp']
            if server.get('command') != 'uv' or arguments[:len(expected)] != expected:
                raise ValueError('Custom Blender MCP command preserved; review it before integration')
            arguments = arguments[len(expected):]
        server.update(command=command, args=arguments)
        config = path, original, (json.dumps(data, indent=2) + '\n').encode()
    atomic(module / 'app/mcp-runtime.json', (json.dumps(receipt, indent=2) + '\n').encode())
    if config is not None:
        path, original, updated = config
        if original != updated:
            fd, name = tempfile.mkstemp(prefix='mcp.json.before-prepared-', dir=plugin)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(original)
            atomic(path, updated)
    return dict(status='prepared', revision=revision, dependency_records=len(metadata), runtime_installs=False)


def run(module, arguments):
    receipt = json.loads((module / 'app/mcp-runtime.json').read_text())
    url, revision = source_pin(receipt['source'])
    if digest(module / 'mcp-requirements.lock') != receipt['requirements_sha256']:
        raise ValueError('Blender MCP dependency declaration changed; prepare it again')
    executable = Path(receipt['executable'])
    provenance = Path(receipt['provenance'])
    if not executable.is_file() or not os.access(executable, os.X_OK) or digest(executable) != receipt['executable_sha256']:
        raise ValueError('Prepared Blender MCP executable is missing or changed')
    if digest(provenance) != receipt['provenance_sha256']:
        raise ValueError('Prepared Blender MCP provenance changed')
    direct = json.loads(provenance.read_text())
    if direct.get('url') != url or direct.get('vcs_info', {}).get('commit_id') != revision:
        raise ValueError('Prepared Blender MCP pin changed')
    if any(digest(Path(row['path'])) != row['sha256'] for row in receipt['dependencies']):
        raise ValueError('Prepared Blender MCP dependency metadata changed')
    os.execv(executable, [str(executable), *arguments])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    install = commands.add_parser('prepare')
    install.add_argument('--module', type=Path, required=True)
    install.add_argument('--runtime', type=Path, required=True)
    install.add_argument('--source', required=True)
    install.add_argument('--plugin', type=Path)
    launch = commands.add_parser('run')
    launch.add_argument('--module', type=Path, required=True)
    launch.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.operation == 'prepare':
            print(json.dumps(prepare(args.module.resolve(), args.runtime, args.source, args.plugin)))
        else:
            arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
            run(args.module, arguments)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('CHECK Blender MCP: ' + str(error) + '; prepare it during dev mod update blender', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
