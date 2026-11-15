"""Expose other modules' declared MCP servers and skills to Hermes only after the owner decides (pull integration).

reconcile (launch.sh, at every "dev hermes" start; install.sh with --no-ask):
  1. reads "ai mod capabilities --json" (what installed module.json files declare; discovery never authorises);
  2. asks the owner once, at an attended start only, about every MCP server or skill not decided yet or whose
     declared command, arguments or effects changed; the answer is kept in ~/.hermes/eva-approved.json;
  3. writes the approved MCP servers (and their toolsets) into Hermes' config through its own CLI, copies approved
     skills into ~/.hermes/skills/modules (marker .module = agent-hermes) and drops whatever is no longer offered
     or approved. A server without declared effects, or one whose id another module already uses, is not offered.
migrate (once, before the first reconcile): remembers what earlier releases enabled without asking - the Blender
  Hermes plugin and the module skills the runner linked; each is approved as soon as its module declares it (that
  module may still be the old release when agent-hermes is updated first), so nothing visible disappears.

eva-approved.json is a remembered decision and an audit trail, not a security boundary: everything here runs as the
same user. The limits stay elsewhere: internet OFF, downloads approved on Windows, declared effects, unknown = not
exposed, and the profile check refuses any configured MCP server that is not approved.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import hermes_profile_check as policy
import hermes_runtime as runtime

ME = 'agent-hermes'
EFFECTS = ('read_only', 'writes', 'gpu', 'network', 'consequential')


def home() -> Path:
    """The selected Hermes home."""
    return Path(os.environ.get('HERMES_HOME', Path.home() / '.hermes'))


def valid_effects(effects: Any) -> bool:
    """Declared, known effects only (writes may carry a scope); anything else is not offered."""
    return (isinstance(effects, list) and bool(effects)
            and all(isinstance(e, str) and e.split(':', 1)[0] in EFFECTS for e in effects))


def offers(capabilities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """What other installed modules declare for agents, keyed mcp:<module>/<id> and skill:<module>/<name>."""
    found: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for module in sorted(capabilities.get('modules', []), key=lambda m: m.get('name', '')):
        name = module.get('name')
        if name == ME or module.get('manifest') != 2:
            continue
        for server in module.get('mcp', []):
            if not valid_effects(server.get('effects')) or server.get('id') in ids:
                print(f'  Hermes: {name} MCP server {server.get("id")} not offered (effects undeclared or id already used)', file=sys.stderr)
                continue
            ids.add(server['id'])
            found[f'mcp:{name}/{server["id"]}'] = dict(kind='mcp', module=name, id=server['id'], command=server['command'],
                                                      args=list(server.get('args') or []), effects=list(server['effects']))
        for skill in module.get('skills', []):
            found[f'skill:{name}/{skill["name"]}'] = dict(kind='skill', module=name, name=skill['name'], path=skill['path'])
    return found


def discover() -> dict[str, dict[str, Any]]:
    """Offers from the installed modules now. A failed discovery offers nothing (fail closed)."""
    try:
        result = subprocess.run(['ai', 'mod', 'capabilities', '--json'], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return offers(json.loads(result.stdout))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    print('  Hermes: module declarations could not be read (ai mod capabilities) - no module tools exposed', file=sys.stderr)
    return {}


def load(path: Path) -> dict[str, Any]:
    """Remembered decisions; an unreadable file decides nothing."""
    try:
        data = json.loads(path.read_text())
        if data.get('schema') == 1 and isinstance(data.get('decisions'), dict):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return dict(schema=1, decisions={})


def save(path: Path, data: dict[str, Any]) -> None:
    """Private, atomic write."""
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.chmod(name, 0o600)
    os.replace(name, path)


def now() -> str:
    """UTC time of a decision."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def describe(offer: dict[str, Any]) -> str:
    """One line for the owner."""
    if offer['kind'] == 'mcp':
        return (f'MCP server "{offer["id"]}" from module {offer["module"]} (effects: {", ".join(offer["effects"])}; '
                f'runs {offer["command"]} {" ".join(offer["args"])})').rstrip()
    return f'skill "{offer["name"]}" from module {offer["module"]} ({offer["path"]})'


def ask(offer: dict[str, Any]) -> bool | None:
    """Ask on the terminal; None when nobody can answer."""
    try:
        with open('/dev/tty', 'r+') as tty:
            tty.write(f'  Let Hermes use the {describe(offer)}? [y/N] ')
            tty.flush()
            return tty.readline().strip().lower() in ('y', 'yes')
    except OSError:
        return None


def hermes(executable: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """The official Hermes CLI in the offline environment."""
    return subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=30, env=runtime.environment())


def get(executable: str, key: str) -> Any:
    """A config value, or None when Hermes reports it unset."""
    result = hermes(executable, 'config', 'get', '--json', key)
    if result.returncode:
        if 'Config key not set' in result.stdout + result.stderr:
            return None
        raise ValueError(f'hermes config get {key} failed')
    return json.loads(result.stdout)


def put(executable: str, key: str, value: Any) -> None:
    """Set a config value and read it back."""
    if hermes(executable, 'config', 'set', key, json.dumps(value)).returncode or get(executable, key) != value:
        raise ValueError(f'hermes config set {key} did not take effect')


def migrate(executable: str, data: dict[str, Any], root: Path) -> None:
    """First run: remember what earlier releases enabled without asking - Hermes plugins and the module skills the
    runner linked. The module that declares it may still be the old release (agent-hermes is often updated first),
    so the memory is kept and carry_over() approves each item once its module declares it."""
    plugins = get(executable, 'plugins.enabled') or []
    skills = root / 'skills/modules'
    linked = sorted(f'{marker.read_text().strip()}/{marker.parent.name}' for marker in skills.glob('*/.module')
                    if marker.is_file() and marker.read_text().strip() != ME) if skills.is_dir() else []
    data['legacy'] = dict(plugins=sorted(p for p in plugins if isinstance(p, str)), skills=linked)
    data['migrated'] = now()


def carry_over(data: dict[str, Any], offered: dict[str, dict[str, Any]]) -> None:
    """Approve what an earlier release had enabled (see migrate), unless the owner has decided about it since."""
    legacy = data.get('legacy') or {}
    for key, offer in sorted(offered.items()):
        if key in data['decisions']:
            continue
        if offer['kind'] == 'mcp' and offer['id'] in legacy.get('plugins', []):
            how = 'migration: Hermes plugin was enabled'
        elif offer['kind'] == 'skill' and f'{offer["module"]}/{offer["name"]}' in legacy.get('skills', []):
            how = 'migration: skill was linked'
        else:
            continue
        data['decisions'][key] = dict(approved=True, offer=offer, decided=now(), how=how)


def apply(executable: str, approved: dict[str, dict[str, Any]], root: Path) -> None:
    """Make Hermes' config and managed skill copies match the approved offers exactly."""
    if get(executable, 'plugins.enabled'):
        put(executable, 'plugins.enabled', [])                      # integrations arrive as MCP servers, not plugins
    servers = {o['id']: dict(command=o['command'], args=o['args']) for o in approved.values() if o['kind'] == 'mcp'}
    current = get(executable, 'mcp_servers') or {}
    for name in sorted(set(current) - set(servers)):
        if hermes(executable, 'config', 'unset', f'mcp_servers.{name}').returncode:
            raise ValueError(f'hermes config unset mcp_servers.{name} failed')
    for name, server in sorted(servers.items()):
        if current.get(name) != server:
            put(executable, f'mcp_servers.{name}', server)
    toolsets = policy.BASE_TOOLSETS + sorted(servers)
    if get(executable, 'platform_toolsets.cli') != toolsets:
        put(executable, 'platform_toolsets.cli', toolsets)
    skills = root / 'skills/modules'
    wanted = {o['name']: Path(o['path']) for o in approved.values() if o['kind'] == 'skill'}
    for folder in sorted(skills.iterdir()) if skills.is_dir() else []:
        marker = folder / '.module'
        if folder.name not in wanted and marker.is_file():         # ours, or an old runner link of a module skill
            shutil.rmtree(folder)
    for name, source in sorted(wanted.items()):
        if not (source / 'SKILL.md').is_file():
            continue
        target = skills / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, symlinks=True)
        (target / '.module').write_text(ME + '\n')


def reconcile(executable: str, attended: bool) -> dict[str, Any]:
    """Decide what is new, then expose exactly what the owner approved. Returns a short summary."""
    root = home()
    path = root / policy.APPROVALS
    offered = discover()
    data = load(path)
    if 'migrated' not in data:
        migrate(executable, data, root)
    carry_over(data, offered)
    asked = skipped = 0
    for key, offer in sorted(offered.items()):
        record = data['decisions'].get(key)
        if record and record.get('offer') == offer:
            continue
        answer = ask(offer) if attended else None
        if answer is None:
            skipped += 1
            print(f'  Hermes: {describe(offer)} is not exposed until you decide at an attended "dev hermes" start', file=sys.stderr)
            continue
        asked += 1
        data['decisions'][key] = dict(approved=answer, offer=offer, decided=now(), how='owner')
    approved = {k: o for k, o in offered.items() if data['decisions'].get(k, {}).get('approved') is True
                and data['decisions'][k].get('offer') == o}
    save(path, data)
    apply(executable, approved, root)
    return dict(status='reconciled', exposed=sorted(approved), asked=asked, undecided=skipped)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('operation', choices=['reconcile'])
    parser.add_argument('--hermes', default='hermes')
    parser.add_argument('--no-ask', action='store_true', help='never ask (installs, unattended starts)')
    args = parser.parse_args(argv)
    attended = not args.no_ask and not os.environ.get('DEV_UNATTENDED')
    try:
        print(json.dumps(reconcile(args.hermes, attended)))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'CHECK Hermes integrations: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
