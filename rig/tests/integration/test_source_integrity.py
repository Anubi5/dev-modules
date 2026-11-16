"""Real temporary Git trees; injected files are harmless text and never imported."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from source_integrity import verify_source


class SourceTests(unittest.TestCase):
    def setUp(self):
        # Retain test fixtures for inspection; TMPDIR/TEMP is set to staging by the caller.
        self.source = Path(tempfile.mkdtemp(prefix="rig-integrity-"))
        self.git("init", "-q")
        (self.source/"src").mkdir()
        (self.source/"src/model.py").write_bytes(b"# pinned, harmless source\n")
        (self.source/".gitignore").write_bytes(b"ignored.py\n__pycache__/\n")
        self.git("add", ".")
        self.git("-c", "user.name=Rig Test", "-c", "user.email=rig-test@example.invalid", "commit", "-qm", "fixture")
        self.revision = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *args):
        return subprocess.check_output(["git", "-c", "core.autocrlf=false", "-C", str(self.source), *args], timeout=20)

    def test_clean_source_has_stable_identity(self):
        identity = verify_source(self.source, self.revision)
        self.assertEqual(len(identity), 64)
        self.assertEqual(verify_source(self.source, self.revision), identity)

    def test_untracked_and_ignored_executables_fail_even_with_clean_git_diff(self):
        for name in ("src/unexpected.py", "ignored.py"):
            with self.subTest(name=name):
                # Use separate fixtures so the failure identifies the actual injected path.
                if name == "ignored.py": self.setUp()
                (self.source/name).write_text("# never executed\n")
                self.git("diff", "--quiet")  # The previous check explicitly passes.
                with self.assertRaisesRegex(RuntimeError, "Unexpected source file"):
                    verify_source(self.source, self.revision)

    def test_staged_changes_cannot_hide_behind_clean_worktree_diff(self):
        (self.source/"src/model.py").write_bytes(b"# changed staged code\n")
        self.git("add", "src/model.py")
        self.git("diff", "--quiet")
        with self.assertRaisesRegex(RuntimeError, "content changed"):
            verify_source(self.source, self.revision)

    def test_missing_wrong_revision_and_empty_namespace_fail(self):
        with self.assertRaisesRegex(RuntimeError, "revision drifted"):
            verify_source(self.source, "0"*40)
        (self.source/"src/namespace_override").mkdir()
        with self.assertRaisesRegex(RuntimeError, "Unexpected source directory"):
            verify_source(self.source, self.revision)

    def test_missing_file_fails(self):
        (self.source/"src/model.py").rename(self.source.parent/(self.source.name+"-retained-model.py"))
        with self.assertRaisesRegex(RuntimeError, "files are missing"):
            verify_source(self.source, self.revision)


if __name__ == "__main__":
    unittest.main()
