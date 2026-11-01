#!/usr/bin/env python3
"""!
@file resolve_models.py
@brief Finds, copies or downloads the model files listed in models.json.

Chat models land in <out>/<id>/ and are recorded in <out>/models.json (id -> file paths).
With --media, the ComfyUI files are placed in <out>/<folder>/ instead.
A file that already exists is reused; with --seed, it is copied from an earlier download
(<seed>/<id>/<file>) before anything is downloaded. Every choice is printed for the log.

Usage:
  resolve_models.py --config models.json --out $MOD/models [--seed /mnt/models-rw/llm]
  resolve_models.py --config models.json --media --out /mnt/models-rw/comfy
"""
import argparse
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


def fetch(repo, filename, dest, seed=None, model_id=None):
    """!
    @brief Makes one file available in dest: reuse it, copy it from the seed folder, or download it.
    @param repo      Repository id.
    @param filename  File name inside the repository.
    @param dest      Target folder.
    @param seed      Optional folder with an earlier download (<seed>/<model_id>/<filename>).
    @param model_id  Model id used for the seed folder layout.
    @return Local path of the file.
    """
    target = os.path.join(dest, filename)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"  {filename}: already present")
        return target
    if seed and model_id:
        src = os.path.join(seed, model_id, filename)
        if os.path.exists(src) and os.path.getsize(src) > 0:
            print(f"  {filename}: copying from {seed}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(src, target + ".part")                       # complete copies only become the real file
            if os.path.getsize(target + ".part") != os.path.getsize(src):
                raise RuntimeError(f"copy of {filename} is incomplete")
            os.replace(target + ".part", target)
            return target
    print(f"  {repo} :: {filename}")
    return download(repo, filename, local_dir=dest)


def chat_models(specs, out, seed=None):
    """!
    @brief Resolves every chat model and writes <out>/models.json with the local file paths.
    @param specs  models.json "chat" list.
    @param out    Folder for the model files (one sub-folder per model id).
    @param seed   Optional folder with an earlier download to copy from.
    @return Number of models that failed.
    """
    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, "models.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    failed = 0
    for spec in specs:
        try:
            print(f"\n[{spec['id']}]")
            old = manifest.get(spec["id"], {})       # offline repair: reuse it only if it matches what models.json asks for now
            have = old.get("files", []) + ([old.get("mmproj")] if spec.get("mmproj") else [])
            if old.get("files") and old.get("quant") == spec["quant"] \
                    and (not spec.get("repo") or old.get("repo") == spec["repo"]) \
                    and all(f and os.path.exists(f) and os.path.getsize(f) > 0 for f in have):
                print(f"  {os.path.basename(old['file'])}: already present ({old.get('repo')}, all {len(old['files'])} part(s))"); continue
            repo = find_repo(spec)
            if not repo:
                print("  no matching repo found"); failed += 1; continue
            files = api.list_repo_files(repo)
            chosen = pick_files(files, spec["quant"])
            if not chosen:
                print(f"  {repo}: no file matching {spec['quant']}"); failed += 1; continue
            dest = os.path.join(out, spec["id"])
            paths = [fetch(repo, f, dest, seed, spec["id"]) for f in chosen]
            entry = {"repo": repo, "file": paths[0], "files": paths, "quant": spec["quant"]}   # every part of split files
            if spec.get("mmproj"):
                mm = sorted((f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")),
                            key=lambda f: (0 if re.search("q8", f, re.I) else 1, len(f)))
                if mm:
                    entry["mmproj"] = fetch(repo, mm[0], dest, seed, spec["id"])
            manifest[spec["id"]] = entry
            json.dump(manifest, open(manifest_path, "w"), indent=2)
        except Exception as e:
            print(f"  {spec['id']} FAILED: {str(e)[:200]}"); failed += 1
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--media", action="store_true", help="ComfyUI files instead of chat models")
    ap.add_argument("--seed", help="earlier download (<seed>/<id>/<file>) to copy from")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    n = media_models(cfg["media"], a.out) if a.media else chat_models(cfg["chat"], a.out, a.seed)
    print(f"\n{n} item(s) failed" if n else "\nall items available")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
