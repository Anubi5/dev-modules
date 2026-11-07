"""Verify the closed, module-owned upstream source tree before importing it."""
import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
import subprocess


def verify_source(source: Path, revision: str) -> str:
    """Return a tree digest after checking every file against pinned Git blobs.

    Unexpected files/directories (including ignored Python/bytecode), symlinks,
    submodules and changed tracked bytes are refused. Git metadata is excluded.
    This is a preflight check, not protection against concurrent privileged writes.
    """
    source = source.resolve(strict=True)
    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(source), *args], timeout=30,
        )
    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise RuntimeError("Installed UniRig source revision drifted")
    expected = {}
    directories = {"."}
    for entry in git("ls-tree", "-r", "-z", "--full-tree", revision).split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        path = PurePosixPath(name.decode())
        if mode not in ("100644", "100755") or kind != "blob" or path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Unsupported upstream source entry: " + str(path))
        expected[str(path)] = blob
        directories.update(str(parent) for parent in path.parents)
    observed = set()
    for root, dirs, files in os.walk(source, followlinks=False):
        if Path(root) == source:
            dirs[:] = [name for name in dirs if name != ".git"]
            files = [name for name in files if name != ".git"]
        for name in dirs + files:
            path = Path(root)/name
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                raise RuntimeError("Unexpected source symlink: " + relative)
            if name in dirs:
                if relative not in directories:
                    raise RuntimeError("Unexpected source directory: " + relative)
                continue
            if relative not in expected:
                raise RuntimeError("Unexpected source file (including ignored files): " + relative)
            if not stat.S_ISREG(path.stat().st_mode):
                raise RuntimeError("Source entry is not a regular file: " + relative)
            content = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if actual != expected[relative]:
                raise RuntimeError("Pinned source content changed: " + relative)
            observed.add(relative)
    if observed != expected.keys():
        raise RuntimeError("Pinned source files are missing")
    digest = hashlib.sha256()
    for name in sorted(expected):
        digest.update(name.encode() + b"\0" + expected[name].encode() + b"\0")
    return digest.hexdigest()
