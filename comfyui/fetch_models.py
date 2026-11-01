#!/usr/bin/env python3
"""!
@file fetch_models.py
@brief Makes every file in models.txt available: present in the read-only Windows folder or the module's folder, else downloaded.
Usage: fetch_models.py <models.txt> <read-only root> <module models root>
"""
import os
import sys

from huggingface_hub import HfApi, hf_hub_download


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
        if any(os.path.exists(os.path.join(root, folder, name)) for root in (ro, own)):
            continue
        try:
            match = [f for f in HfApi().list_repo_files(repo) if f == want or f.endswith("/" + want)][0]
            print(f"  downloading {repo} :: {match}", flush=True)
            path = hf_hub_download(repo, match, local_dir=os.path.join(own, "_dl"))
            os.makedirs(os.path.join(own, folder), exist_ok=True)
            os.replace(path, os.path.join(own, folder, name))
        except Exception as e:
            print(f"  {repo} {name}: {e}"); failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
