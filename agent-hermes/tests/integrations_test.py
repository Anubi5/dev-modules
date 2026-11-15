"""Pull integration offline: only owner-approved module MCP servers and skills reach Hermes (fake CLIs, temp home)."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hermes_profile_check as policy
import integrations


class Fakes:
    """Hermes' config CLI (get/set/unset on a dict) and "ai mod capabilities --json"."""

    def __init__(self, root):
        self.root = Path(root)
        self.config = {'plugins': {'enabled': []}, 'platform_toolsets': {'cli': list(policy.BASE_TOOLSETS)}}
        self.modules = []
        self.discovery_fails = False

    def module(self, name, mcp=(), skills=()):
        folder = self.root / 'modules' / name
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
            target[leaf] = json.loads(command[4])
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

    def test_migration_keeps_what_earlier_releases_enabled(self):
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])], skills=['blender-sandbox'])
        self.fake.config['plugins']['enabled'] = ['blender']
        old = self.home / 'skills/modules/blender-sandbox'
        old.mkdir(parents=True); (old / '.module').write_text('blender\n'); (old / 'SKILL.md').write_text('old')
        result = self.reconcile()
        self.assertEqual(result['exposed'], ['mcp:blender/blender', 'skill:blender/blender-sandbox'])
        self.assertEqual(self.fake.config['plugins']['enabled'], [])
        self.assertEqual(self.servers()['blender']['args'], ['--transport', 'stdio'])
        self.assertEqual(self.fake.config['platform_toolsets']['cli'], policy.BASE_TOOLSETS + ['blender'])
        self.assertEqual((old / '.module').read_text().strip(), 'agent-hermes')
        self.assertIn('name: blender-sandbox', (old / 'SKILL.md').read_text())
        stored = json.loads((self.home / policy.APPROVALS).read_text())
        self.assertEqual(stored['decisions']['mcp:blender/blender']['how'], 'migration: Hermes plugin was enabled')
        self.assertEqual(oct((self.home / policy.APPROVALS).stat().st_mode & 0o777)[-3:], '600') if os.name == 'posix' else None

    def test_migration_waits_for_a_module_still_on_the_old_release(self):
        # dev update updates agent-hermes first: Blender is still the release with the Hermes plugin, no declaration
        self.fake.config['plugins']['enabled'] = ['blender']
        old = self.home / 'skills/modules/blender-sandbox'
        old.mkdir(parents=True); (old / '.module').write_text('blender\n'); (old / 'SKILL.md').write_text('old')
        first = self.reconcile()
        self.assertEqual(first['exposed'], [])
        self.assertEqual(self.fake.config['plugins']['enabled'], [])
        self.fake.module('blender', mcp=[('blender', ['writes', 'gpu'])], skills=['blender-sandbox'])   # Blender updated
        result = self.reconcile()                                  # unattended: nobody is asked, the old state returns
        self.assertEqual((result['exposed'], result['asked']), (['mcp:blender/blender', 'skill:blender/blender-sandbox'], 0))
        self.assertEqual(self.servers()['blender']['args'], ['--transport', 'stdio'])
        self.assertIn('name: blender-sandbox', (old / 'SKILL.md').read_text())

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
        self.assertEqual(sorted(integrations.discover()), ['mcp:blender/blender'])

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


if __name__ == '__main__':
    unittest.main()
