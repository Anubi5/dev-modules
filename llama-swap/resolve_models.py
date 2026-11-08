#!/usr/bin/env python3
"""!
@file resolve_models.py
@brief Finds or downloads the chat model files listed in models.json - they live on D:, never inside the distro.

Chat model files live in D:\\AI\\models\\llm\\<id>\\ (read-only at /models/llm inside). Downloads and the one-time move
of older in-distro copies need D: writable at /models-rw, which only Windows turns on ("dev mod add|update").
The record of which files each model uses is <state>/models.json (id -> file paths). Every choice is printed.
With --media, the ComfyUI files are placed in <out>/<folder>/ instead (older setup path).

Usage:
  resolve_models.py --config models.json --state $MOD/models --ro /models/llm [--rw /models-rw/llm]
  resolve_models.py --config models.json --media --out /models-rw/comfy
"""
import argparse
import copy
import json
import os
import re
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True)            # show progress in the live log
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")       # plain HTTPS with resume copes better with flaky links
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import HfApi, hf_hub_download as _hf_download  # noqa: E402

api = HfApi()


def download(repo, filename, **kw):
    """!
    @brief Downloads one file from Hugging Face, retrying; partial files resume on the next attempt.
    @param repo      Repository id, e.g. "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".
    @param filename  File path inside the repository.
    @param kw        Passed on to hf_hub_download (e.g. local_dir).
    @return Local path of the downloaded file.
    """
    for attempt in range(1, 9):
        try:
            return _hf_download(repo, filename, **kw)
        except Exception as e:  # network errors: wait and retry
            print(f"  attempt {attempt} failed: {str(e)[:160]}")
            time.sleep(min(30 * attempt, 180))
    raise RuntimeError(f"giving up on {repo}/{filename}")


def find_repo(spec):
    """!
    @brief Returns the repository for a chat model: the pinned 'repo' (never replaced by a search if it is unavailable),
           or - only when no repo is pinned - the most-downloaded search match.
    @param spec  One entry of models.json "chat".
    @return Repository id, or None if nothing matches.
    """
    if spec.get("repo"):
        try:
            api.model_info(spec["repo"])
            return spec["repo"]
        except Exception as e:
            print(f"  pinned repo {spec['repo']} unavailable ({e}) - not substituting another one")
            return None
    if not spec.get("search"):
        return None
    for m in api.list_models(search=spec["search"], sort="downloads", limit=100):
        if all(t in m.id.lower() for t in spec.get("must", [])):
            return m.id
    return None


def pick_files(files, quants):
    """!
    @brief Chooses the model file(s) for the first preferred quantisation that exists (all parts of split files).
    @param files   All file names in the repository.
    @param quants  Preferred quantisation names, best first.
    @return List of file names (empty if none match).
    """
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    for q in quants:
        # the quantisation must be the file name's last tag (repo names like "...-q4_0-unquantized" must not match)
        hit = [f for f in ggufs if re.search(r"[-_.]" + re.escape(q) + r"(-\d{5}-of-\d{5})?\.gguf$", f, re.I)]
        if hit:
            parts = sorted(f for f in hit if re.search(r"-\d{5}-of-\d{5}", f))
            return parts or [sorted(hit, key=len)[0]]
    return []


def same_file(a, b):
    """! @brief True if both files exist with the same non-zero size (model files are never edited in place). """
    return os.path.exists(a) and os.path.exists(b) and os.path.getsize(a) == os.path.getsize(b) > 0


def same_content(a, b):
    """! @brief True only if both files have the same SHA-256 (used before deleting a copy - size alone is not proof). """
    if not same_file(a, b):
        return False
    import hashlib

    def h(p):
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                d.update(block)
        return d.hexdigest()
    return h(a) == h(b)


def move_to_d(src, dst, keep=False):
    """!
    @brief Moves an older in-distro copy to D: - copy, then compare the SHA-256 of the copy on D: with the original,
           then delete the in-distro copy (unless keep).
    @param src  File inside the distro.  @param dst  Path under the writable D: mount.  @param keep  Keep src.
    """
    keep or print(f"  {os.path.basename(src)}: moving to D: ({os.path.getsize(src) >> 20} MB)")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".part")
    if not same_content(src, dst + ".part"):
        os.remove(dst + ".part"); raise RuntimeError(f"copy of {os.path.basename(src)} to D: does not match - original kept")
    os.replace(dst + ".part", dst)
    keep or os.remove(src)


def place(rel, ro, rw, legacy, repo=None, local=False):
    """!
    @brief Makes one file available on D: and returns the path to load it from.
           On D: already -> use it (and delete a leftover in-distro copy). Only in the distro -> move it to D: when
           D: is writable, else keep using it for now. Missing -> download straight to D: (needs D: writable).
           local=True (models.json "local_copy"): D: keeps the original, and a copy inside the distro is loaded -
           for a model that reads its file while it runs (MoE experts in RAM), which is much slower from D:.
           Rule for local copies: D: is the authority - a local copy that differs in size is replaced from D:.
           Two copies of a NON-local model must be identical (SHA-256); otherwise the model fails (no guessing).
    @param rel     "<id>/<file in repo>".  @param ro  Read-only D: root (/models/llm).
    @param rw      Writable D: root (/models-rw/llm) or None.  @param legacy  In-distro root ($MOD/models).
    @param repo    Repository to download from; None = offline check only.  @param local  Keep a copy in the distro.
    @return Path to load the file from.
    """
    on_d, old = os.path.join(ro, rel), os.path.join(legacy, rel)
    have_d = os.path.exists(on_d) and os.path.getsize(on_d) > 0
    have_old = os.path.exists(old) and os.path.getsize(old) > 0
    if local and have_old and (same_file(old, on_d) or (not have_d and not rw)):
        return old
    if have_d:
        if not local and os.path.exists(old):             # two copies: same content (hash) or a conflict
            if same_content(old, on_d):
                os.remove(old); print(f"  {os.path.basename(rel)}: removed the duplicate inside the distro (same file on D:)")
            else:                                         # which one is right is unknown: fail this model, don't guess
                raise RuntimeError(f"{os.path.basename(rel)}: the copy inside the distro ({old}) differs from the one on D: "
                                   f"- both kept; delete the wrong one, then run dev mod update llama-swap again")
        if not local:
            return on_d
    elif have_old:
        if not rw:
            print(f"  {os.path.basename(rel)}: still inside the distro only - run 'dev mod update llama-swap' from Windows to move it to D:")
            return old
        if local:                                        # original to D:, keep the distro copy
            print(f"  {os.path.basename(rel)}: copying the original to D:")
            move_to_d(old, os.path.join(rw, rel), keep=True)
            return old
        move_to_d(old, os.path.join(rw, rel)); return on_d
    else:
        if not repo:
            raise RuntimeError(f"{rel} is missing")
        if not rw:
            raise RuntimeError(f"{rel} is missing - downloads go to D:\\AI\\models, run 'dev mod update llama-swap' from Windows")
        print(f"  {repo} :: {os.path.basename(rel)} -> D:")
        model_id, filename = rel.split("/", 1)
        download(repo, filename, local_dir=os.path.join(rw, model_id))
        shutil.rmtree(os.path.join(rw, model_id, ".cache"), ignore_errors=True)   # Hugging Face bookkeeping
        if not local:
            return on_d
    print(f"  {os.path.basename(rel)}: copying into the distro (local_copy: loads from the distro's own disk)")
    os.makedirs(os.path.dirname(old), exist_ok=True)
    shutil.copyfile(on_d, old + ".part")
    if os.path.getsize(old + ".part") != os.path.getsize(on_d):
        os.remove(old + ".part"); raise RuntimeError(f"copy of {os.path.basename(rel)} is incomplete")
    os.replace(old + ".part", old)
    return old


def upgrade_record(old, spec):
    """!
    @brief Older records only list the first file: add every split part found next to it and the quantisation,
           but only when that file is the first-choice quantisation (else the online lookup decides).
    @param old  Record from models.json (may be empty).  @param spec  The models.json "chat" entry.
    @return The record, upgraded when that is certain.
    """
    if not old.get("file") or old.get("files"):
        return old
    first = spec["quant"][0]
    if not re.search(r"[-_.]" + re.escape(first) + r"(-\d{5}-of-\d{5})?\.gguf$", old["file"], re.I):
        return old
    m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", old["file"])
    parts = [old["file"]] if not m else \
        [re.sub(r"-\d{5}-of-", f"-{i:05d}-of-", old["file"]) for i in range(1, int(m.group(1)) + 1)]
    return dict(old, files=parts, quant=spec["quant"])


def save_manifest(path, manifest):
    with open(path+'.pending','w') as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(path+'.pending',path)


def chat_models(specs, state, ro, rw=None):
    """!
    @brief Resolves every chat model and writes <state>/models.json with the paths to load each file from.
    @param specs  models.json "chat" list.
    @param state  In-distro folder for the record (and, from older setups, model copies that are moved to D:).
    @param ro     Read-only D: folder (/models/llm).  @param rw  Writable D: folder (/models-rw/llm) or None.
    @return Number of models that failed.
    """
    os.makedirs(state, exist_ok=True)
    manifest_path = os.path.join(state, "models.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as stream: manifest = json.load(stream)
    failed = 0
    by_id = {spec['id']: spec for spec in specs}
    if len(by_id) != len(specs):
        print('CHECK duplicate chat model IDs; nothing resolved')
        return 1
    resolved = set()
    # Weight variants may reference one ordinary model, irrespective of list order.
    for spec in sorted(specs, key=lambda row: bool(row.get('weights_from'))):
        try:
            print(f"\n[{spec['id']}]")
            source = spec.get('weights_from')
            if source:
                if source not in resolved or by_id[source].get('weights_from'):
                    raise RuntimeError('weights_from must reference a successfully resolved ordinary model')
                entry = copy.deepcopy(manifest[source])
                paths = entry['files'] + ([entry['mmproj']] if entry.get('mmproj') else [])
                root = os.path.realpath(ro)
                if any(not os.path.isfile(p) or os.path.commonpath([root, os.path.realpath(p)]) != root for p in paths):
                    raise RuntimeError('Shared weights must already exist in the canonical read-only model store')
                entry['weights_from'] = source
                manifest[spec['id']] = entry
                save_manifest(manifest_path, manifest)
                resolved.add(spec['id'])
                print('  reuses the existing files of '+source+' (no copy or download)')
                continue
            old = upgrade_record(manifest.get(spec["id"], {}), spec)

            def rel(p):   # "<id>/<file>" of a recorded path, whether it is on D: or inside the distro
                for root in (state, ro, rw or ro):
                    if p.startswith(root.rstrip("/") + "/"):
                        return os.path.relpath(p, root)
                return os.path.join(spec["id"], os.path.basename(p))
            # offline: reuse the record if it matches what models.json asks for now (moving files to D: on the way)
            if old.get("files") and old.get("quant") == spec["quant"] \
                    and (not spec.get("repo") or old.get("repo") == spec["repo"]):
                try:
                    local = bool(spec.get("local_copy"))
                    entry = dict(old, files=[place(rel(f), ro, rw, state, local=local) for f in old["files"]])
                    entry["file"] = entry["files"][0]
                    if spec.get("mmproj") and old.get("mmproj"):
                        entry["mmproj"] = place(rel(old["mmproj"]), ro, rw, state)   # small, read once
                    if not spec.get("mmproj") or entry.get("mmproj"):
                        manifest[spec["id"]] = entry
                        resolved.add(spec['id'])
                        save_manifest(manifest_path, manifest)
                        print(f"  {os.path.basename(entry['file'])}: ready ({old.get('repo')}, {len(entry['files'])} part(s))")
                        continue
                except RuntimeError as e:
                    if "is missing" not in str(e):
                        raise                             # e.g. two different copies: report, don't look further
                    # a recorded file is gone: look it up again below
            repo = find_repo(spec)
            if not repo:
                print("  no matching repo found"); failed += 1; continue
            files = api.list_repo_files(repo)
            chosen = pick_files(files, spec["quant"])
            if not chosen:
                print(f"  {repo}: no file matching {spec['quant']}"); failed += 1; continue
            paths = [place(f"{spec['id']}/{f}", ro, rw, state, repo, bool(spec.get("local_copy"))) for f in chosen]
            entry = {"repo": repo, "file": paths[0], "files": paths, "quant": spec["quant"]}   # every part of split files
            if spec.get("mmproj"):
                mm = sorted((f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")),
                            key=lambda f: (0 if re.search("q8", f, re.I) else 1, len(f)))
                if mm:
                    entry["mmproj"] = place(f"{spec['id']}/{mm[0]}", ro, rw, state, repo)
            manifest[spec["id"]] = entry
            resolved.add(spec['id'])
            save_manifest(manifest_path, manifest)
        except Exception as e:
            print(f"  {spec['id']} FAILED: {str(e)[:200]}"); failed += 1
    for d, _, _ in sorted(os.walk(state), reverse=True):          # empty model folders left after the move
        if d != state and not os.listdir(d):
            os.rmdir(d)
    return failed


def media_models(specs, out):
    """!
    @brief Places every ComfyUI file in <out>/<folder>/, downloading only what is missing.
    @param specs  models.json "media" list.
    @param out    ComfyUI model root (e.g. <ModelsWin>/comfy).
    @return Number of files that failed.
    """
    failed = 0
    staging = os.path.join(out, "_dl")
    for spec in specs:
        dest = os.path.join(out, spec["folder"])
        os.makedirs(dest, exist_ok=True)
        if os.path.exists(os.path.join(dest, spec["file"])):
            print(f"  {spec['file']}: already present"); continue
        try:
            match = [f for f in api.list_repo_files(spec["repo"]) if f.endswith(spec["file"])]
            if not match:
                print(f"  {spec['repo']}: {spec['file']} not found"); failed += 1; continue
            print(f"  {spec['repo']} :: {match[0]} -> {spec['folder']}")
            path = download(spec["repo"], match[0], local_dir=staging)
            os.replace(path, os.path.join(dest, os.path.basename(path)))
        except Exception as e:
            print(f"  {spec['repo']}: {e}"); failed += 1
    shutil.rmtree(staging, ignore_errors=True)
    return failed


def main():
    """!
    @brief Command-line entry point; exit code 0 only if every item is available.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="models.json")
    ap.add_argument("--out", help="--media: ComfyUI model root")
    ap.add_argument("--media", action="store_true", help="ComfyUI files instead of chat models")
    ap.add_argument("--state", help="in-distro folder for the record (models.json)")
    ap.add_argument("--ro", default="/models/llm", help="chat models on D:, read-only")
    ap.add_argument("--rw", help="the same folder writable (only during dev mod add|update from Windows)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    rw = a.rw if a.rw and os.path.isdir(a.rw) and os.access(a.rw, os.W_OK) else None
    n = media_models(cfg["media"], a.out) if a.media else chat_models(cfg["chat"], a.state, a.ro, rw)
    print(f"\n{n} item(s) failed" if n else "\nall items available")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
