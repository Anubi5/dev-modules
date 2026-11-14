"""The managed SOUL section never consumes the owner's text before or after it."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('soul_section', HERE / 'soul_section.py')
soul = importlib.util.module_from_spec(spec); spec.loader.exec_module(soul)
TEMPLATE = (HERE / 'hermes-local-SOUL.md').read_text()
LINES = TEMPLATE.splitlines(keepends=True)
BODY = LINES[1:-1]
LEGACY = ''.join(BODY).replace('in the Windows model store (/models', 'under X:\\Some\\models (/models')


class SoulSectionTests(unittest.TestCase):
    def test_added_after_own_text(self):
        text, action = soul.refresh(TEMPLATE, 'My persona.\n')
        self.assertEqual(action, 'added'); self.assertTrue(text.startswith('My persona.\n')); self.assertIn(TEMPLATE, text)

    def test_marked_section_refreshed_with_text_before_and_after(self):
        short = LINES[0] + BODY[0] + 'old short body\n' + LINES[-1]
        text, action = soul.refresh(TEMPLATE, 'Before.\n' + short + 'After, mine.\nMore of mine.\n')
        self.assertEqual(action, 'refreshed')
        self.assertTrue(text.startswith('Before.\n' + TEMPLATE)); self.assertTrue(text.endswith('After, mine.\nMore of mine.\n'))

    def test_unmarked_legacy_section_migrated_exactly(self):
        text, action = soul.refresh(TEMPLATE, 'Before.\n' + LEGACY + 'After, mine.\n')
        self.assertEqual((action, text), ('migrated', 'Before.\n' + TEMPLATE + 'After, mine.\n'))

    def test_shorter_or_edited_legacy_section_is_left_untouched(self):
        for old in (''.join(BODY[:3]) + 'my own line right after\n', LEGACY.replace('Never search', 'Please never search')):
            with self.subTest(old=old[:30]):
                original = 'Before.\n' + old + 'After, mine.\n'
                self.assertEqual(soul.refresh(TEMPLATE, original), (original, 'review'))

    def test_unterminated_or_duplicated_markers_left_untouched(self):
        for original in (LINES[0] + 'mine\n', TEMPLATE + TEMPLATE):
            with self.subTest(original=original[:20]):
                self.assertEqual(soul.refresh(TEMPLATE, original)[0], original)

    def test_more_malformed_markers(self):
        begin, end = LINES[0], LINES[-1]
        cases = {
            'end marker only': ('Mine.\n' + end + 'Still mine.\n', 'added'),
            'two differing begin markers': (begin + 'a\n' + end + begin + 'b\n' + end, 'review'),
            'begin without its end, end earlier': (end + 'Mine.\n' + begin + 'Mine too.\n', 'review'),
        }
        for name, (original, action) in cases.items():
            with self.subTest(name):
                text, got = soul.refresh(TEMPLATE, original)
                self.assertEqual(got, action)
                self.assertTrue(text.startswith(original))   # the owner's text is never removed or reordered

    def test_current_section_unchanged(self):
        self.assertEqual(soul.refresh(TEMPLATE, 'x\n' + TEMPLATE), ('x\n' + TEMPLATE, 'unchanged'))


if __name__ == '__main__':
    unittest.main()
