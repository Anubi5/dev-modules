#!/usr/bin/env python3
"""!
@file fetch_models.py
@brief Makes every file in models.txt available on D: (read-only inside at /models/comfy): present there, moved there
       from an older in-distro copy, or downloaded there. Writing to D: needs the writable mount (/models-rw), which
       only Windows turns on while it runs "dev mod add|update"; without it, missing files are reported, not fetched.
Usage: fetch_models.py <models.txt> <read-only root> <writable root or -> <state dir> [<old in-distro root>]
       (helpers.txt: read-only root /nonexistent, writable root = state dir = the node's own folder)
"""
import hashlib
import os
import shutil
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


def load_record(rec):
    """! @brief Reads the .hashes record: "<sha256> <size> <mtime_ns> <path>" (older lines: "<sha256> <path>"). @return {path: (sha, size, mtime)} """
    known = {}
    if os.path.exists(rec):
        for line in open(rec):
            f = line.rstrip("\n").split(" ", 3)
            if len(f) == 4 and f[1].isdigit() and f[2].isdigit():
                known[f[3]] = (f[0], int(f[1]), int(f[2]))
            elif len(f) >= 2:
                known[" ".join(f[1:])] = (f[0], -1, -1)
    return known


def save_record(rec, known):
    """! @brief Writes the .hashes record atomically. @param rec record path @param known {path: (sha, size, mtime)} """
    with open(rec + ".new", "w") as f:
        f.writelines(f"{v[0]} {v[1]} {v[2]} {k}\n" for k, v in known.items())
    os.replace(rec + ".new", rec)


def check_hash(path, own, pinned=None, key=None):
    """!
    @brief Integrity check for a model file, present or just downloaded.
           Pinned files (sha256: in the list) must match the pin. Unpinned pickle-based files (can run code when
           loaded) are recorded on first sight and refused if they change later. Unpinned safetensors are not checked.
           A file that was already checked and has not changed (same size and time) is not hashed again.
    @param path  The model file (absolute).  @param own  Writable folder holding the .hashes record.
    @param pinned  Expected SHA-256 from the list, or None.
    @param key  Path the record is kept under (the read-only path ComfyUI loads), default path.
    @return True if the file may be used.
    """
    if not pinned and not path.endswith(CODE_FORMATS):
        return True
    key = key or path
    rec = os.path.join(own, ".hashes")
    known, st = load_record(rec), os.stat(path)
    old = known.get(key)
    if old and old[1:] == (st.st_size, st.st_mtime_ns) and (not pinned or old[0].lower() == pinned.lower()):
        return True                                   # checked before, unchanged
    h = sha256(path)
    want = (pinned or (old[0] if old else h)).lower()
    if h.lower() == want:
        known[key] = (h, st.st_size, st.st_mtime_ns)
        save_record(rec, known)
        return True
    why = "does not match its pinned checksum" if pinned else "changed since it was first fetched"
    if os.access(os.path.dirname(path), os.W_OK):
        os.replace(path, path + ".quarantined")      # so nothing loads it
        print(f"  CHECK {path} {why} - moved aside as {os.path.basename(path)}.quarantined (delete it to fetch again)")
    else:
        print(f"  CHECK {path} {why} - it is in the read-only Windows models folder: replace or delete it there"
              f" (D:\\AI\\models\\...). ComfyUI could still load it, so the install stops here.")
    return False


def move_to_d(src, dst, key, own, pinned):
    """!
    @brief Moves an older in-distro copy to D: - copies it while hashing, checks the hash against the pin or the
           earlier record, keeps the record under the new path, then deletes the in-distro copy.
    @param src  In-distro file.  @param dst  Path under the writable D: mount.  @param key  Read-only path of dst.
    @param own  Folder with the .hashes record.  @param pinned  SHA-256 from the list, or None.
    @return True if moved.
    """
    rec = os.path.join(own, ".hashes")
    known = load_record(rec)
    want = (pinned or (known.get(os.path.abspath(src)) or ("",))[0]).lower()
    print(f"  {os.path.basename(src)}: moving to D: ({os.path.getsize(src) >> 20} MB)", flush=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = hashlib.sha256()
    with open(src, "rb") as i, open(dst + ".part", "wb") as o:
        for block in iter(lambda: i.read(1 << 20), b""):
            o.write(block); h.update(block)
    if (want and h.hexdigest() != want) or os.path.getsize(dst + ".part") != os.path.getsize(src):
        os.remove(dst + ".part")
        print(f"  CHECK {os.path.basename(src)}: the copy on D: does not match - left inside the distro")
        return False
    os.replace(dst + ".part", dst)
    st = os.stat(dst)
    known.pop(os.path.abspath(src), None)
    known[key] = (h.hexdigest(), st.st_size, st.st_mtime_ns)
    save_record(rec, known)
    os.remove(src)
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
    lst, ro, rw, own = sys.argv[1:5]
    old_root = sys.argv[5] if len(sys.argv) > 5 else None
    rw = rw if rw != "-" and os.path.isdir(rw) and os.access(rw, os.W_OK) else None
    os.makedirs(own, exist_ok=True)
    failed = 0
    for line in open(lst):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        repo, want, folder = parts[:3]
        sha = next((x[7:] for x in parts[3:] if x.startswith("sha256:")), None)
        optional = "optional" in parts[3:]                # missing = that model is just not available, not an error
        name = next((x for x in parts[3:] if not x.startswith("sha256:") and x != "optional"), os.path.basename(want))
        on_d = os.path.join(ro, folder, name)
        old = os.path.join(old_root, folder, name) if old_root else None
        if old and os.path.exists(old) and not os.path.exists(on_d) and rw:
            move_to_d(old, os.path.join(rw, folder, name), on_d, own, sha)
        if old and os.path.exists(old) and os.path.exists(on_d) and os.path.getsize(old) == os.path.getsize(on_d):
            if sha256(old) == sha256(on_d):                   # size alone is no proof: compare the content
                os.remove(old); print(f"  {folder}/{name}: removed the duplicate inside the distro (same file on D:)")
            else:
                print(f"  CHECK {folder}/{name}: the copy inside the distro differs from D: - both kept")
        have = [p for p in (on_d, old) if p and os.path.exists(p)]
        if have:
            print(f"  {folder}/{name}: present{'' if have[0] == on_d else ' (still inside the distro - run dev mod update comfyui from Windows to move it to D:)'}", flush=True)
            failed += 0 if check_hash(os.path.abspath(have[0]), own, sha) else 1
            continue
        if not rw:
            msg = f"{folder}/{name} is missing - downloads go to D:\\AI\\models, run 'dev mod update comfyui' from Windows"
            print(f"  (optional) {msg}" if optional else f"  CHECK {msg}"); failed += 0 if optional else 1
            continue
        dst = os.path.join(rw, folder, name)
        if repo.startswith("https://"):
            try:
                if not sha:
                    raise RuntimeError("direct downloads need a sha256:<hex> in the list")
                print(f"  downloading {name} from {repo.split('/')[2]} (checksum pinned)", flush=True)
                fetch_url(repo, dst, sha)
                check_hash(dst, own, sha, key=os.path.abspath(on_d))   # record it as checked
            except Exception as e:
                print(f"  CHECK {name}: {e}"); failed += 1
            continue
        try:
            match = [f for f in HfApi().list_repo_files(repo) if f == want or f.endswith("/" + want)][0]
            print(f"  downloading {repo} :: {match}", flush=True)
            path = hf_hub_download(repo, match, local_dir=os.path.join(rw, "_dl"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(path, dst)
            if not check_hash(dst, own, sha, key=os.path.abspath(on_d)):
                failed += 1
        except Exception as e:
            gated = "gated" in type(e).__name__.lower() or "401" in str(e) or "403" in str(e)
            why = (f"needs a Hugging Face login: accept its licence on huggingface.co/{repo} and put the file into "
                   f"D:\\AI\\models\\comfy\\{folder.replace('/', chr(92))}\\{name}") if gated else str(e)[:300]
            if optional:
                print(f"  (optional) {folder}/{name} not available - {why}")
            else:
                print(f"  CHECK {repo} {name}: {why}"); failed += 1
    if rw:
        shutil.rmtree(os.path.join(rw, "_dl"), ignore_errors=True)   # Hugging Face staging folder
    if old_root and os.path.isdir(old_root):
        for d, _, _ in sorted(os.walk(old_root), reverse=True):      # empty folders left after the move
            if d != old_root and not os.listdir(d):
                os.rmdir(d)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
