"""Owner-attended module integration, preserving unrelated Hermes configuration.

Offers are inert discovery. Exact owner decisions bind MCP commands/arguments/effects
and skill contents. Legacy plugin IDs and marker files never authorize new execution.
Only unchanged recorded EVA entries may be replaced or removed; collisions require
owner resolution. Plugins and unrelated MCP servers, CLI toolsets and skills are
preserved. The profile checker may refuse those entries without deleting them.
A private bounded journal restores configuration, skills and decisions after failure
or restart. Same-UID execution remains an operational trust boundary, not isolation.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
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
    from collections import Counter
    counts = Counter(s.get('id') for m in capabilities.get('modules', []) for s in m.get('mcp', []))
    ids: set[str] = set()
    for module in sorted(capabilities.get('modules', []), key=lambda m: m.get('name', '')):
        name = module.get('name')
        if name == ME or module.get('manifest') != 2:
            continue
        for server in module.get('mcp', []):
            if not valid_effects(server.get('effects')) or counts[server.get('id')] != 1:
                print(f'  Hermes: {name} MCP server {server.get("id")} not offered (effects undeclared or id already used)', file=sys.stderr)
                continue
            ids.add(server['id'])
            found[f'mcp:{name}/{server["id"]}'] = dict(kind='mcp', module=name, id=server['id'], command=server['command'],
                                                      args=list(server.get('args') or []), effects=list(server['effects']),
                                                      command_sha256=hashlib.sha256(Path(server['command']).read_bytes()).hexdigest(),
                                                      implementation_sha256=module.get('implementation_sha256'))
        for skill in module.get('skills', []):
            found[f'skill:{name}/{skill["name"]}'] = dict(kind='skill', module=name, name=skill['name'], path=skill['path'], sha256=skill_digest(Path(skill['path'])))
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
    """Unreadable existing owner decisions are preserved and block reconciliation."""
    if not path.exists():
        return dict(schema=1, decisions={})
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('decisions'), dict):
        raise ValueError('Invalid Hermes approval record; preserved for owner review')
    return data


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
        # Buffered read/write mode requires seeking; a terminal is not seekable.
        with open('/dev/tty', 'r') as reader, open('/dev/tty', 'w') as writer:
            writer.write(f'  Let Hermes use the {describe(offer)}? [y/N] ')
            writer.flush()
            return reader.readline().strip().lower() in ('y', 'yes')
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


def sync_servers(executable: str, wanted: dict[str, Any]) -> None:
    """Use official leaf operations: setting an empty mapping does not remove Hermes config children."""
    import re
    current = get(executable, 'mcp_servers') or {}
    if not isinstance(current, dict):
        raise ValueError('Invalid Hermes MCP configuration')
    for name in sorted(set(current) | set(wanted)):
        if current.get(name) == wanted.get(name) and (name in current) == (name in wanted):
            continue
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,40}', name):
            raise ValueError(f'Cannot safely change MCP config key: {name}')
        key = 'mcp_servers.' + name
        if name not in wanted:
            if hermes(executable, 'config', 'unset', key).returncode or name in (get(executable, 'mcp_servers') or {}):
                raise ValueError(f'hermes config unset {key} did not take effect')
        else:
            put(executable, key, wanted[name])
    if (get(executable, 'mcp_servers') or {}) != wanted:
        raise ValueError('Hermes MCP leaf reconciliation did not take effect')


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
    """Legacy identifiers are evidence of prior use, never approval of a new execution definition."""
    for record in data['decisions'].values():
        if isinstance(record, dict) and str(record.get('how', '')).startswith('migration:'):
            record['approved'] = False
            record['how'] = 'legacy identity requires owner review'


def skill_digest(path: Path) -> str:
    """Fingerprint a plain skill tree; never copy links to executable content outside the offer."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'Skill is not a plain directory: {path}')
    digest = hashlib.sha256()
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError(f'Skill contains a symlink: {item}')
        if item.is_file() and item.name != '.module':
            digest.update(item.relative_to(path).as_posix().encode() + b'\0' + item.read_bytes() + b'\0')
    if not (path / 'SKILL.md').is_file():
        raise ValueError(f'Skill has no SKILL.md: {path}')
    return digest.hexdigest()


def apply(executable: str, approved: dict[str, dict[str, Any]], root: Path,
          previous: dict[str, Any] | None = None, *, validate_only: bool = False) -> dict[str, Any]:
    """Reconcile only unchanged, recorded EVA entries; collisions fail before any mutation."""
    previous = previous or {}
    current = get(executable, 'mcp_servers') or {}
    toolsets = get(executable, 'platform_toolsets.cli')
    if not isinstance(current, dict) or not isinstance(toolsets, list):
        raise ValueError('Hermes MCP/toolset configuration has an unexpected type')
    servers = {o['id']: dict(command=o['command'], args=o['args']) for o in approved.values() if o['kind'] == 'mcp'}
    old_servers = previous.get('servers', {})
    for name, old in old_servers.items():
        if name in current and current[name] != old:
            raise ValueError(f'Owner-modified MCP server collision: {name}')
    for name in servers:
        if name in current and name not in old_servers:
            raise ValueError(f'Unowned MCP server collision: {name}')
    merged = {k: v for k, v in current.items() if k not in old_servers}
    merged.update(servers)
    kept = [t for t in toolsets if t not in previous.get('toolsets', [])]
    added = [name for name in sorted(servers) if name not in kept]
    wanted = {}
    for offer in approved.values():
        if offer['kind'] != 'skill':
            continue
        name = offer['name']
        if name in wanted:
            raise ValueError(f'Ambiguous skill name: {name}')
        source = Path(offer['path'])
        fingerprint=skill_digest(source)
        if fingerprint!=offer.get('sha256'):
            raise ValueError(f'Skill offer changed during review: {name}')
        wanted[name] = (source, fingerprint)
    skills = root / 'skills/modules'
    if skills.is_symlink() or (root / 'skills').is_symlink():
        raise ValueError('Managed skill parent is a symlink')
    old_skills = previous.get('skills', {})
    for name in set(old_skills) | set(wanted):
        target = skills / name
        if target.exists() or target.is_symlink():
            if name not in old_skills or skill_digest(target) != old_skills[name]:
                raise ValueError(f'Unowned or owner-modified skill collision: {name}')
    if validate_only:
        return {}
    # The caller has already journalled these config keys and every affected target.
    if merged != current:
        sync_servers(executable, merged)
    if kept + added != toolsets:
        put(executable, 'platform_toolsets.cli', kept + added)
    for name in sorted(set(old_skills) | set(wanted)):
        target = skills / name
        if target.exists():
            shutil.rmtree(target)
        if name in wanted:
            source, fingerprint = wanted[name]
            shutil.copytree(source, target)
            (target / '.module').write_text(ME + '\n')
            if skill_digest(target)!=fingerprint:
                raise ValueError(f'Skill changed while copying: {name}')
    return dict(servers=servers, toolsets=added, skills={n: d for n, (_, d) in wanted.items()})


@contextmanager
def locked(root: Path):
    """Serialize reconciliations; the OS releases this advisory lock after process death."""
    with (root / '.eva-integration.lock').open('a+b') as stream:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        else:
            import msvcrt
            stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if os.name == 'posix':
                fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def recover(executable: str, root: Path) -> None:
    """Roll back an interrupted reconciliation before permitting launch; retain evidence on failure."""
    journal = root / 'eva-integration-pending.json'
    if not journal.exists():
        return
    data = json.loads(journal.read_text())
    backup = root / data['backup']
    if backup.parent != root or not backup.name.startswith('.eva-integration-'):
        raise ValueError('Invalid integration recovery path')
    for key, value in data['config'].items():
        if key == 'mcp_servers':
            sync_servers(executable, value or {})
            if value is None and hermes(executable, 'config', 'unset', key).returncode:
                raise ValueError('Integration rollback failed: mcp_servers')
            continue
        if value is None:
            if hermes(executable, 'config', 'unset', key).returncode:
                raise ValueError(f'Integration rollback failed: {key}')
        else:
            put(executable, key, value)
    for name, existed in data['skills'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid integration recovery skill')
        target = root / 'skills/modules' / name
        if target.is_symlink():
            raise ValueError(f'Integration recovery refuses symlink: {target}')
        if target.exists():
            shutil.rmtree(target)
        if existed:
            shutil.copytree(backup / name, target)
    approval = root / policy.APPROVALS
    if data['approvals'] is None:
        approval.unlink(missing_ok=True)
    else:
        save(approval, data['approvals'])
    journal.unlink()
    shutil.rmtree(backup)


def verify_offers(approved: dict[str, dict[str, Any]]) -> None:
    """Do not commit an offer changed by a concurrent lifecycle operation or during an owner prompt."""
    if not approved:
        return
    current=discover()
    for key,offer in approved.items():
        if current.get(key)!=offer:
            raise ValueError(f'Integration offer changed during reconciliation: {key}; review it again')


def commit_integration(executable: str, approved: dict[str, dict[str, Any]], root: Path,
                       data: dict[str, Any], before: dict[str, Any] | None) -> None:
    """Journal the bounded write set, update it, then commit approvals; failure restores the previous state."""
    verify_offers(approved)
    previous = (before or {}).get('managed', {})
    names = set(previous.get('skills', {})) | {o['name'] for o in approved.values() if o['kind'] == 'skill'}
    # Validate names before using them as filesystem paths, including remembered records.
    if any(not isinstance(n, str) or Path(n).name != n or n in ('.', '..') or '/' in n or '\\' in n for n in names):
        raise ValueError('Invalid integration skill name')
    apply(executable, approved, root, previous, validate_only=True)
    backup = Path(tempfile.mkdtemp(prefix='.eva-integration-', dir=root))
    journal = root / 'eva-integration-pending.json'
    snapshot = dict(config={key: get(executable, key) for key in ('mcp_servers', 'platform_toolsets.cli')},
                    skills={}, approvals=before, backup=backup.name)
    try:
        for name in names:
            target = root / 'skills/modules' / name
            if target.is_symlink() or (root / 'skills/modules').is_symlink() or (root / 'skills').is_symlink():
                raise ValueError(f'Integration target is a symlink: {target}')
            snapshot['skills'][name] = target.exists()
            if target.exists():
                skill_digest(target)
                shutil.copytree(target, backup / name)
        save(journal, snapshot)
        try:
            data['managed'] = apply(executable, approved, root, previous)
            verify_offers(approved)
            save(root / policy.APPROVALS, data)
        except BaseException:
            recover(executable, root)
            raise
        journal.unlink()  # Commit point: approvals and configuration agree.
    finally:
        if not journal.exists() and backup.exists():
            shutil.rmtree(backup)


def _reconcile(executable: str, attended: bool) -> dict[str, Any]:
    """Decide what is new, then expose exactly what the owner approved. Returns a short summary."""
    root = home()
    path = root / policy.APPROVALS
    offered = discover()
    before = load(path) if path.exists() else None
    data = deepcopy(before) if before else dict(schema=1, decisions={})
    if 'migrated' not in data:
        migrate(executable, data, root)
    carry_over(data, offered)
    asked = skipped = 0
    for key, offer in sorted(offered.items()):
        record = data['decisions'].get(key)
        if record and record.get('offer') == offer and record.get('how') != 'legacy identity requires owner review':
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
    data['active'] = sorted(approved)
    commit_integration(executable, approved, root, data, before)
    return dict(status='reconciled', exposed=sorted(approved), asked=asked, undecided=skipped)


def reconcile(executable: str, attended: bool) -> dict[str, Any]:
    """Recover interrupted writes and serialize the complete owner-decision/configuration update."""
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    with locked(root):
        recover(executable, root)
        return _reconcile(executable, attended)


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
