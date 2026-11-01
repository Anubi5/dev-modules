#!/usr/bin/env python3
"""!
@file fetch_models.py
@brief Makes every file in models.txt available: present in the read-only Windows folder or the module's folder, else downloaded.
Usage: fetch_models.py <models.txt> <read-only root> <module models root>
"""
import hashlib
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")    # the Xet downloader can hang after a file is complete
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


def fetch_url(url, dest, sha):
    """!
    @brief Downloads a file from a direct URL (e.g. CivitAI) to dest, only if its SHA-256 matches the pinned one.
    @param url  Download URL.  @param dest  Final path.  @param sha  Expected SHA-256 (hex).
    @return None; raises on failure (the partial file is removed).
    """
    import urllib.error
    import urllib.request
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dev-modules/1.0"})
    h, done, last = hashlib.sha256(), 0, 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block); h.update(block); done += len(block)
                if done - last >= 256 << 20:   # progress every 256 MB
                    last = done; print(f"    {done >> 20} MB{f' of {total >> 20} MB' if total else ''}", flush=True)
    except urllib.error.HTTPError as e:
        os.path.exists(part) and os.remove(part)
        if e.code in (401, 403):
            raise RuntimeError(f"the site wants a login for this file - download it in your browser and put it in "
                               f"D:\\AI\\models\\comfy\\{os.path.basename(os.path.dirname(dest))}") from None
        raise
    if h.hexdigest().lower() != sha.lower():
        os.remove(part)
        raise RuntimeError(f"checksum mismatch (expected {sha[:12]}..., got {h.hexdigest()[:12]}...) - file discarded")
    os.replace(part, dest)


def main():
    """!
    @brief Reads "<repo> <file or path in repo> <folder> [save as]" lines (Hugging Face) and
           "<https URL> <file name> <folder> sha256:<hex>" lines (direct download, checksum required) and
           downloads only what is missing. @return exit code
    """
    lst, ro, own = sys.argv[1:4]
    failed = 0
    for line in open(lst):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        repo, want, folder = parts[:3]
        sha = next((x[7:] for x in parts[3:] if x.startswith("sha256:")), None)
        name = next((x for x in parts[3:] if not x.startswith("sha256:")), os.path.basename(want))
        have = [os.path.join(root, folder, name) for root in (ro, own) if os.path.exists(os.path.join(root, folder, name))]
        if have:
            print(f"  {folder}/{name}: present", flush=True)
            failed += 0 if check_hash(os.path.abspath(have[0]), own) else 1
            continue
        if repo.startswith("https://"):
            try:
                if not sha:
                    raise RuntimeError("direct downloads need a sha256:<hex> in the list")
                print(f"  downloading {name} from {repo.split('/')[2]} (checksum pinned)", flush=True)
                fetch_url(repo, os.path.join(own, folder, name), sha)
            except Exception as e:
                print(f"  CHECK {name}: {e}"); failed += 1
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
