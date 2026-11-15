"""Pull integration offline: only owner-approved module MCP servers and skills reach Hermes (fake CLIs, temp home)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hermes_profile_check as policy
import integrations


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires a real Linux terminal')
class TerminalDecisionTests(unittest.TestCase):
    """Exercise the actual terminal streams, rather than mocking owner answers."""

    def test_real_terminal_accepts_approval_and_refusal(self):
        import pty
        import select
        import signal
        for answer, expected in (('yes', True), ('n', False)):
            with self.subTest(answer=answer):
                pid, fd = pty.fork()
                if pid == 0:
                    try:
                        value = integrations.ask(dict(kind='skill', module='fixture', name='fixture', path='/fixture'))
                        os._exit(0 if value is expected else 1)
                    except BaseException:
                        os._exit(2)
                reaped = False
                try:
                    output = b''
                    deadline = time.monotonic() + 5
                    while b'[y/N] ' not in output and time.monotonic() < deadline:
                        if select.select([fd], [], [], 0.1)[0]:
                            try:
                                output += os.read(fd, 4096)
                            except OSError:
                                break
                    self.assertIn(b'[y/N] ', output)
                    os.write(fd, (answer + '\n').encode())
                    while time.monotonic() < deadline:
                        done, status = os.waitpid(pid, os.WNOHANG)
                        if done:
                            reaped = True
                            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
                            break
                        time.sleep(0.01)
                    self.assertTrue(reaped, 'terminal decision timed out')
                finally:
                    if not reaped:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                    os.close(fd)


class Fakes:
    """Hermes' config CLI (get/set/unset on a dict) and "ai mod capabilities --json"."""

    def __init__(self, root):
        self.root = Path(root)
        self.config = {'plugins': {'enabled': []}, 'platform_toolsets': {'cli': list(policy.BASE_TOOLSETS)}}
        self.modules = []
        self.discovery_fails = False

    def module(self, name, mcp=(), skills=()):
        folder = self.root / 'modules' / name
        (folder / 'bin').mkdir(parents=True,exist_ok=True)
        for identifier, effects in mcp:
            (folder / 'bin' / identifier).write_text('#!/bin/sh\nexit 0\n')
        for skill in skills:
            (folder / 'skills' / skill).mkdir(parents=True, exist_ok=True)
            (folder / 'skills' / skill / 'SKILL.md').write_text(f'---\nname: {skill}\n---\n')
        self.modules.append(dict(name=name, manifest=2, capabilities=[], requires=[], optional=[], effects={},
                                 mcp=[dict(id=i, command=f'{folder}/bin/{i}', args=['--transport', 'stdio'], effects=e) for i, e in mcp],
                                 skills=[dict(name=s, path=str(folder / 'skills' / s)) for s in skills]))

    def node(self, key):
        value = self.config
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(key)
            value = value[part]
        return value

    def run(self, command, **kwargs):
        if command[:3] == ['ai', 'mod', 'capabilities']:
            return SimpleNamespace(returncode=1 if self.discovery_fails else 0, stdout=json.dumps(dict(schema=1, modules=self.modules)), stderr='')
        verb = command[2]
        if verb == 'get':
            try:
                return SimpleNamespace(returncode=0, stdout=json.dumps(self.node(command[4])), stderr='')
            except KeyError:
                return SimpleNamespace(returncode=1, stdout='', stderr='Config key not set: ' + command[4])
        *parents, leaf = command[3].split('.')
        target = self.config
        for part in parents:
            target = target.setdefault(part, {})
        if verb == 'set':
            value = json.loads(command[4])
            if command[3] == 'mcp_servers' and isinstance(value, dict):
                target.setdefault(leaf, {}).update(value)  # official CLI deep-merges root maps
            else:
                target[leaf] = value
        else:
            target.pop(leaf, None)
        return SimpleNamespace(returncode=0, stdout='', stderr='')


class IntegrationTests(unittest.TestCase):
    """Each test reconciles against fakes in a disposable Hermes home."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name) / 'hermes'
        self.home.mkdir()
        self.fake = Fakes(self.temp.name)
        patches = [patch.dict(os.environ, HERMES_HOME=str(self.home)), patch.object(integrations.subprocess, 'run', side_effect=self.fake.run),
                   patch.object(integrations.runtime, 'environment', return_value={})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def reconcile(self, answers=None):
        """answers: list of True/False for attended prompts; None = unattended."""
        if answers is None:
            return integrations.reconcile('hermes', attended=False)
        with patch.object(integrations, 'ask', side_effect=list(answers)):
            return integrations.reconcile('hermes', attended=True)

    def servers(self):
        return self.fake.config.get('mcp_servers', {})

    def test_legacy_identifiers_do_not_approve_changed_execution(self):
        self.fake.config['plugins']['enabled'] = ['blender', 'owner-plugin']
        self.reconcile()
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])])
        self.assertEqual(self.reconcile()['exposed'], [])
        self.assertEqual(self.fake.config['plugins']['enabled'], ['blender', 'owner-plugin'])
        self.assertEqual(self.reconcile([True])['exposed'], ['mcp:blender/blender'])

    def test_unowned_skill_is_preserved_even_with_legacy_marker(self):
        self.fake.module('blender', skills=['blender-sandbox'])
        old = self.home / 'skills/modules/blender-sandbox'
        old.mkdir(parents=True)
        (old / '.module').write_text('blender\n')
        (old / 'SKILL.md').write_text('owner text')
        self.reconcile()
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.reconcile([True])
        self.assertEqual((old / 'SKILL.md').read_text(), 'owner text')
        self.assertEqual((old / '.module').read_text(), 'blender\n')

    def test_owner_decision_wins_over_the_old_release(self):
        self.fake.config['plugins']['enabled'] = ['blender']
        self.reconcile()
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])])
        stored = json.loads((self.home / policy.APPROVALS).read_text())
        stored['decisions']['mcp:blender/blender'] = dict(approved=False, offer=integrations.discover()['mcp:blender/blender'],
                                                          decided='2026-11-15T00:00:00Z', how='owner')
        (self.home / policy.APPROVALS).write_text(json.dumps(stored))
        self.assertEqual(self.reconcile()['exposed'], [])
        self.assertNotIn('blender', self.servers())

    def test_undecided_is_not_exposed_without_an_attended_start(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])], skills=['blender-sandbox'])
        result = self.reconcile()
        self.assertEqual((result['exposed'], result['undecided']), ([], 2))
        self.assertNotIn('blender', self.servers())
        self.assertFalse((self.home / 'skills/modules/blender-sandbox').exists())

    def test_owner_decides_once(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])], skills=['blender-sandbox'])
        self.reconcile()                                           # migration point passes with nothing enabled
        result = self.reconcile([True, False])                     # mcp first (sorted keys), then the skill
        self.assertEqual(result['exposed'], ['mcp:blender/blender'])
        self.assertIn('blender', self.servers())
        again = self.reconcile([])                                 # nothing new: nobody is asked
        self.assertEqual((again['asked'], again['exposed']), (0, ['mcp:blender/blender']))

    def test_changed_declaration_is_asked_again_and_dropped_meanwhile(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])])
        self.reconcile(); self.reconcile([True])
        self.fake.modules[0]['mcp'][0]['effects'] = ['writes', 'gpu', 'network']
        result = self.reconcile()                                  # unattended: the new declaration is not approved
        self.assertEqual(result['exposed'], [])
        self.assertNotIn('blender', self.servers())
        self.assertEqual(self.fake.config['platform_toolsets']['cli'], policy.BASE_TOOLSETS)

    def test_removed_provider_is_dropped(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])], skills=['blender-sandbox'])
        self.reconcile(); self.reconcile([True, True])
        self.fake.modules.clear()
        self.reconcile()
        self.assertEqual(self.servers(), {})
        self.assertFalse((self.home / 'skills/modules/blender-sandbox').exists())

    def test_failed_discovery_and_undeclared_effects_expose_nothing(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])])
        self.reconcile(); self.reconcile([True])
        self.fake.discovery_fails = True
        self.assertEqual(self.reconcile()['exposed'], [])
        self.assertNotIn('blender', self.servers())
        self.fake.discovery_fails = False
        self.fake.module('odd', mcp=[('odd', []), ('blender', ['read_only'])])   # no effects; id taken by another module
        self.assertEqual(sorted(integrations.discover()), [])  # neither ambiguous offer wins

    def test_profile_accepts_approved_servers_only(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])])
        self.reconcile(); self.reconcile([True])
        approved = policy.approved_servers(self.home)
        values = dict(policy.EXPECTED, **{'skills.disabled': sorted(policy.DISABLED_SKILLS), 'mcp_servers': self.servers(),
                                          'platform_toolsets.cli': self.fake.config['platform_toolsets']['cli']})
        values.update({key: None for key in policy.FALLBACK_KEYS})
        self.assertEqual(policy.evaluate(values, approved=approved)['errors'], [])
        self.assertTrue(policy.evaluate(values)['errors'])        # nothing approved: the configured server is refused
        changed = dict(values, mcp_servers={'blender': dict(self.servers()['blender'], args=['--other'])})
        self.assertIn('Unapproved MCP server: blender', policy.evaluate(changed, approved=approved)['errors'])
        plugin = dict(values, **{'plugins.enabled': ['blender']})
        self.assertTrue(policy.evaluate(plugin, approved=approved)['errors'])

    def test_unrelated_configuration_survives_approve_and_remove(self):
        self.fake.config['plugins']['enabled'] = ['personal']
        self.fake.config['mcp_servers'] = {'personal': {'command': 'mine', 'args': [], 'env': {'X': 'Y'}}}
        self.fake.config['platform_toolsets']['cli'].append('personal')
        self.fake.module('blender', mcp=[('blender', ['writes'])])
        self.reconcile([True])
        self.fake.modules.clear()
        self.reconcile()
        self.assertEqual(self.servers(), {'personal': {'command': 'mine', 'args': [], 'env': {'X': 'Y'}}})
        self.assertEqual(self.fake.config['plugins']['enabled'], ['personal'])
        self.assertEqual(self.fake.config['platform_toolsets']['cli'], policy.BASE_TOOLSETS + ['personal'])

    def test_mcp_collision_fails_before_mutation(self):
        from copy import deepcopy
        self.fake.module('blender', mcp=[('blender', ['writes'])])
        self.fake.config['mcp_servers'] = {'blender': {'command': 'owner', 'args': []}}
        before = deepcopy(self.fake.config)
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.reconcile([True])
        self.assertEqual(self.fake.config, before)
        self.assertFalse((self.home / policy.APPROVALS).exists())

    def test_owner_modified_managed_entry_is_preserved(self):
        self.fake.module('blender', mcp=[('blender', ['writes'])])
        self.reconcile([True])
        self.servers()['blender']['args'] = ['owner-change']
        self.fake.modules.clear()
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.reconcile()
        self.assertEqual(self.servers()['blender']['args'], ['owner-change'])

    def test_skill_content_change_needs_new_approval(self):
        self.fake.module('blender', skills=['example'])
        self.reconcile([True])
        source = Path(self.fake.modules[0]['skills'][0]['path'])
        (source / 'SKILL.md').write_text('new executable instructions')
        self.assertEqual(self.reconcile()['exposed'], [])
        self.assertFalse((self.home / 'skills/modules/example').exists())
        self.assertEqual(self.reconcile([True])['exposed'], ['skill:blender/example'])

    def test_partial_write_rolls_back_config_skills_and_decisions(self):
        from copy import deepcopy
        self.fake.module('blender', mcp=[('blender', ['writes'])], skills=['example'])
        self.reconcile()
        before = deepcopy(self.fake.config)
        decisions = (self.home / policy.APPROVALS).read_bytes()
        original = integrations.put
        failed = False
        def failing(executable, key, value):
            nonlocal failed
            if key == 'platform_toolsets.cli' and not failed:
                failed = True
                raise ValueError('injected write failure')
            return original(executable, key, value)
        with patch.object(integrations, 'put', side_effect=failing), self.assertRaisesRegex(ValueError, 'injected'):
            self.reconcile([True, True])
        self.assertEqual(self.fake.config, before)
        self.assertEqual((self.home / policy.APPROVALS).read_bytes(), decisions)
        self.assertFalse((self.home / 'eva-integration-pending.json').exists())
        self.assertFalse((self.home / 'skills/modules/example').exists())
        self.assertEqual(len(self.reconcile([True, True])['exposed']), 2)

    def test_server_removal_uses_leaf_unset_with_deep_merging_cli(self):
        self.fake.module('blender',mcp=[('blender',['writes'])])
        self.fake.config['mcp_servers']={'personal':{'command':'owner','args':[]}}
        self.reconcile([True])
        self.fake.modules.clear();self.reconcile()
        self.assertEqual(self.servers(),{'personal':{'command':'owner','args':[]}})

    def test_historical_approval_cannot_authorize_restored_unowned_configuration(self):
        from copy import deepcopy
        self.fake.module('blender',mcp=[('blender',['writes'])])
        self.reconcile([True])
        old=deepcopy(self.servers())
        self.fake.modules.clear();self.reconcile()
        self.assertEqual(policy.approved_servers(self.home),{})
        # An unrelated restoration of the old config does not restore current managed ownership.
        self.fake.config['mcp_servers']=old
        self.fake.config['platform_toolsets']['cli']=policy.BASE_TOOLSETS+['blender']
        self.reconcile()
        values=dict(policy.EXPECTED,**{'skills.disabled':sorted(policy.DISABLED_SKILLS),
                    'mcp_servers':old,'platform_toolsets.cli':policy.BASE_TOOLSETS+['blender']})
        values.update({key:None for key in policy.FALLBACK_KEYS})
        self.assertIn('Unapproved MCP server: blender',policy.evaluate(values,approved=policy.approved_servers(self.home))['errors'])

    def test_changed_mcp_implementation_requires_new_decision(self):
        self.fake.module('blender', mcp=[('blender', ['writes'])])
        self.reconcile([True])
        Path(self.fake.modules[0]['mcp'][0]['command']).write_text('changed execution')
        self.assertEqual(self.reconcile()['exposed'], [])
        self.assertNotIn('blender', self.servers())
        self.assertEqual(self.reconcile([True])['exposed'], ['mcp:blender/blender'])

    def test_malformed_approval_file_is_preserved(self):
        path=self.home/policy.APPROVALS;path.write_text('{invalid owner record')
        with self.assertRaises(ValueError):self.reconcile()
        self.assertEqual(path.read_text(),'{invalid owner record')

    def test_offer_changed_during_owner_prompt_is_not_activated(self):
        self.fake.module('blender',mcp=[('blender',['writes'])])
        command=Path(self.fake.modules[0]['mcp'][0]['command'])
        def approve_then_change(offer):
            command.write_text('new unreviewed execution')
            return True
        with patch.object(integrations,'ask',side_effect=approve_then_change),self.assertRaisesRegex(ValueError,'changed during'):
            integrations.reconcile('hermes',attended=True)
        self.assertNotIn('blender',self.servers())
        self.assertFalse((self.home/policy.APPROVALS).exists())

    def test_skill_changed_during_copy_is_rolled_back(self):
        self.fake.module('blender',skills=['example'])
        copy=integrations.shutil.copytree
        def changed(source,target,*args,**kwargs):
            result=copy(source,target,*args,**kwargs)
            (Path(target)/'SKILL.md').write_text('unreviewed concurrent content')
            return result
        with patch.object(integrations.shutil,'copytree',side_effect=changed),self.assertRaisesRegex(ValueError,'changed while copying'):
            self.reconcile([True])
        self.assertFalse((self.home/'skills/modules/example').exists())
        self.assertFalse((self.home/policy.APPROVALS).exists())

    def test_approval_commit_failure_restores_previous_skill_copy(self):
        self.fake.module('blender',skills=['example'])
        self.reconcile([True])
        target=self.home/'skills/modules/example/SKILL.md'
        original=target.read_bytes()
        decisions=(self.home/policy.APPROVALS).read_bytes()
        source=Path(self.fake.modules[0]['skills'][0]['path'])/'SKILL.md'
        source.write_text('new reviewed instructions')
        save=integrations.save;failed=False
        def fail_commit(path,data):
            nonlocal failed
            if path.name==policy.APPROVALS and not failed:
                failed=True
                raise OSError('injected approval commit failure')
            return save(path,data)
        with patch.object(integrations,'save',side_effect=fail_commit),self.assertRaisesRegex(OSError,'injected'):
            self.reconcile([True])
        self.assertEqual(target.read_bytes(),original)
        self.assertEqual((self.home/policy.APPROVALS).read_bytes(),decisions)
        self.assertEqual(self.reconcile([True])['exposed'],['skill:blender/example'])
        self.assertEqual(target.read_text(),'new reviewed instructions')

    def test_restart_recovers_journal_before_reconciliation(self):
        backup = self.home / '.eva-integration-interrupted'
        backup.mkdir()
        integrations.save(self.home / 'eva-integration-pending.json', dict(
            backup=backup.name, config={'mcp_servers': {}, 'platform_toolsets.cli': policy.BASE_TOOLSETS},
            skills={}, approvals=None))
        self.fake.config['mcp_servers'] = {'uncommitted': {'command': 'wrong'}}
        self.reconcile()
        self.assertEqual(self.servers(), {})
        self.assertFalse(backup.exists())

    def test_legacy_migrated_decision_requires_owner_again(self):
        self.fake.module('blender', mcp=[('blender', ['writes'])])
        offer = integrations.discover()['mcp:blender/blender']
        integrations.save(self.home / policy.APPROVALS, dict(schema=1, migrated='old', decisions={
            'mcp:blender/blender': dict(approved=True, offer=offer, how='migration: Hermes plugin was enabled')}))
        self.assertEqual(self.reconcile()['exposed'], [])
        self.assertEqual(self.reconcile([True])['exposed'], ['mcp:blender/blender'])


if __name__ == '__main__':
    unittest.main()
