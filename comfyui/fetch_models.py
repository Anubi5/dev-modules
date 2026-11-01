#!/usr/bin/env python3
"""!
@file fetch_models.py
@brief Makes every file in models.txt available: present in the read-only Windows folder or the module's folder, else downloaded.
Usage: fetch_models.py <models.txt> <read-only root> <module models root>
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

CODE_FORMATS = (".pt", ".pth", ".bin", ".ckpt", ".pkl")   # pickle-based: can run code when loaded


def sha256(path):
    """! @brief SHA-256 of a file. @param path file @return hex digest """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_hash(path, own):
    """!
    @brief Pickle-based files: records the hash the first time, refuses the file if it changes later.
    @param path  The model file (absolute).  @param own  Writable folder holding the .hashes record.
    @return True if the file may be used.
    """
    if not path.endswith(CODE_FORMATS):
        return True
    rec = os.path.join(own, ".hashes")
    known = {}
    if os.path.exists(rec):
        for line in open(rec):
            if line.strip():
                digest, name = line.rstrip("\n").split(" ", 1)
                known[name] = digest
    h = sha256(path)
    if path not in known:
        with open(rec, "a") as f:
            f.write(f"{h} {path}\n")
        return True
    if known[path] != h:
        if os.access(os.path.dirname(path), os.W_OK):
            os.replace(path, path + ".quarantined")   # so nothing loads it
        print(f"  CHECK {path} changed since it was first fetched (hash mismatch) - moved aside as .quarantined;"
              f" if the change is expected, delete its line from {rec}")
        return False
    return True


def main():
    """! @brief Reads "<repo> <file or path in repo> <folder> [save as]" lines and downloads only what is missing. @return exit code """
    lst, ro, own = sys.argv[1:4]
    failed = 0
    for line in open(lst):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        repo, want, folder = parts[:3]
        name = parts[3] if len(parts) > 3 else os.path.basename(want)
        have = [os.path.join(root, folder, name) for root in (ro, own) if os.path.exists(os.path.join(root, folder, name))]
        if have:
            failed += 0 if check_hash(os.path.abspath(have[0]), own) else 1
            continue
        try:
            match = [f for f in HfApi().list_repo_files(repo) if f == want or f.endswith("/" + want)][0]
            print(f"  downloading {repo} :: {match}", flush=True)
            path = hf_hub_download(repo, match, local_dir=os.path.join(own, "_dl"))
            os.makedirs(os.path.join(own, folder), exist_ok=True)
            os.replace(path, os.path.join(own, folder, name))
            if not check_hash(os.path.abspath(os.path.join(own, folder, name)), own):
                failed += 1
        except Exception as e:
            print(f"  {repo} {name}: {e}"); failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
