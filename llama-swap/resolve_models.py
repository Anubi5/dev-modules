#!/usr/bin/env python3
"""!
@file resolve_models.py
@brief Reuses existing chat weights; new files require reviewed pins and explicit owner confirmation.

Chat model files live in D:\\AI\\models\\llm\\<id>\\ (read-only at /models/llm inside).
Legacy copies are preserved; no automatic migration, replacement or retirement.
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
import hashlib
import tempfile

sys.stdout.reconfigure(line_buffering=True)            # show progress in the live log
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")       # plain HTTPS with resume copes better with flaky links
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from pathlib import Path
api = None


def _hf_download(*args, **kwargs):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(*args, **kwargs)


def declarations(path):
    return json.loads(Path(path).read_text())


def download_plan(config=None, root=None):
    config=Path(config or Path(__file__).with_name('models.json'))
    root=Path(root or os.environ.get('MODELS_MOUNT','/models'));files=[]
    for spec in declarations(config)['chat']:
        for row in spec.get('pinned_files',[]):
            rel='llm/'+spec['id']+'/'+row['file'];target=root/rel
            if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe declared model path')
            if target.is_file():continue
            policy=spec.get('download_policy',{});sha=row.get('sha256');size=row.get('bytes')
            eligible=bool(re.fullmatch('[a-fA-F0-9]{64}',sha or '')) and bool(re.fullmatch('[a-fA-F0-9]{40}',spec.get('revision',''))) and bool(spec.get('repo')) and type(size) is int and size>0 and policy.get('approved') is True and bool(policy.get('license')) and not policy.get('gated')
            files.append(dict(path=rel,bytes=size,source=spec.get('repo','')+'@'+spec.get('revision','')+'/'+row['file'],licence=policy.get('license'),sha256=sha.lower() if sha else None,eligible=eligible,reason=None if eligible else 'Manual copy: incomplete or restricted declaration'))
    return dict(schema=1,files=files)


def approved_download(sha):
    """Only argument-derived approvals from the Windows plan may authorize bytes."""
    if os.environ.get('DEV_MODEL_OPERATION') != 'approved' or os.environ.get('DEV_UNATTENDED'):
        return False
    try:
        approved = json.loads(os.environ.get('DEV_APPROVED_SHA256', '[]'))
    except (ValueError, TypeError):
        return False
    return (isinstance(approved, list)
            and all(isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) for value in approved)
            and isinstance(sha, str) and sha.lower() in approved)


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
    global api
    if api is None:
        from huggingface_hub import HfApi
        api=HfApi()
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


def place(rel, ro, rw, legacy, repo=None, local=False, declaration=None):
    """Preserve all existing weights. Missing pinned files need individual owner consent."""
    if os.path.isabs(rel) or '..' in Path(rel).parts or ':' in rel or '\\' in rel:
        raise RuntimeError('Unsafe model path')
    on_d,old=os.path.join(ro,rel),os.path.join(legacy,rel)
    have_d=os.path.isfile(on_d) and os.path.getsize(on_d)>0
    have_old=os.path.isfile(old) and os.path.getsize(old)>0
    if have_d:
        if have_old and not same_content(old,on_d):raise RuntimeError('Legacy and canonical weights differ; both preserved')
        return on_d
    if have_old:
        print('  Legacy weight preserved; copy manually to D: for '+rel)
        return old
    if not repo or not rw or os.environ.get('DEV_MODEL_OPERATION')!='approved' or os.environ.get('DEV_UNATTENDED'):
        raise RuntimeError(rel+' is missing; run dev mod update llama-swap for confirmed pinned acquisition, or copy manually')
    if declaration is None:raise RuntimeError('Missing declared pins/licence; copy manually')
    spec,row=declaration;policy=spec.get('download_policy',{})
    if (policy.get('approved') is not True or not policy.get('license') or policy.get('gated')
            or not re.fullmatch('[a-fA-F0-9]{40}',spec.get('revision',''))
            or not re.fullmatch('[a-fA-F0-9]{64}',row.get('sha256',''))
            or type(row.get('bytes')) is not int or row['bytes']<=0):
        raise RuntimeError('Missing reviewed size/hash/licence/revision; gated models stay manual')
    filename=row['file']
    print(f"  Missing pinned file: {rel}; bytes={row['bytes']}; source={repo}@{spec['revision']}/{filename}; licence={policy['license']}; SHA-256={row['sha256']}",flush=True)
    if not approved_download(row['sha256']):
        raise RuntimeError('SHA-256 not approved; download refused')
    os.makedirs(rw,exist_ok=True)
    staging=tempfile.mkdtemp(prefix='.model-stage-',dir=rw)
    downloaded=download(repo,filename,revision=spec['revision'],local_dir=staging,token=False)
    digest=hashlib.sha256()
    with open(downloaded,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    if digest.hexdigest().lower()!=row['sha256'].lower() or os.path.getsize(downloaded)!=row['bytes']:
        raise RuntimeError('Downloaded file differs from declared pins; staging preserved')
    destination=os.path.join(rw,rel)
    os.makedirs(os.path.dirname(destination),exist_ok=True)
    os.link(downloaded,destination) # fails atomically if any destination already exists
    os.unlink(downloaded) # published link retains the verified weight
    return on_d


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
            # Declared pins are checked even when an older serving record can be reused.
            for row in spec.get('pinned_files',[]):
                for root in (ro,state):
                    path=Path(root)/spec['id']/row['file']
                    if not path.resolve().is_relative_to(Path(root).resolve()):raise RuntimeError('Unsafe pinned model path')
                    if path.is_file():
                        digest=hashlib.sha256()
                        with path.open('rb') as stream:
                            for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
                        if path.stat().st_size!=row['bytes'] or digest.hexdigest().lower()!=row['sha256'].lower():
                            raise RuntimeError('Declared model pin mismatch; existing file preserved')
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
            declared=spec.get('pinned_files',[])
            if not declared or not spec.get('repo'):
                raise RuntimeError('Missing local model or declared pinned files; use dev mod update llama-swap after metadata review, or copy manually')
            repo=spec['repo'];models=[row for row in declared if row.get('role','model')=='model']
            projectors=[row for row in declared if row.get('role')=='mmproj']
            if not models or (spec.get('mmproj') and len(projectors)!=1):raise RuntimeError('Incomplete pinned file declaration')
            paths=[place(spec['id']+'/'+row['file'],ro,rw,state,repo,declaration=(spec,row)) for row in models]
            entry=dict(repo=repo,file=paths[0],files=paths,quant=spec['quant'])
            if spec.get('mmproj'):
                row=projectors[0];entry['mmproj']=place(spec['id']+'/'+row['file'],ro,rw,state,repo,declaration=(spec,row))
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
    """Legacy media entry point follows the same no-clobber, confirmed pin policy."""
    failed=0
    for spec in specs:
        try:place(spec['folder']+'/'+spec['file'],out,out,out,spec.get('repo'),declaration=(spec,spec))
        except Exception as error:print('CHECK '+str(error));failed+=1
    return failed


def main():
    """!
    @brief Command-line entry point; exit code 0 only if every item is available.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name('models.json')), help="models.json")
    ap.add_argument('--plan',action='store_true');ap.add_argument('--json',action='store_true');ap.add_argument('--models-root')
    ap.add_argument("--out", help="--media: ComfyUI model root")
    ap.add_argument("--media", action="store_true", help="ComfyUI files instead of chat models")
    ap.add_argument("--state", help="in-distro folder for the record (models.json)")
    ap.add_argument("--ro", default="/models/llm", help="chat models on D:, read-only")
    ap.add_argument("--rw", help="the same folder writable (only during dev mod add|update from Windows)")
    a = ap.parse_args()
    if a.plan or a.json:
        if not(a.plan and a.json):ap.error('Use --plan --json together')
        print(json.dumps(download_plan(a.config,a.models_root)));return
    cfg = declarations(a.config)
    rw = a.rw if a.rw and os.path.isdir(a.rw) and os.access(a.rw, os.W_OK) else None
    n = media_models(cfg["media"], a.out) if a.media else chat_models(cfg["chat"], a.state, a.ro, rw)
    print(f"\n{n} item(s) failed" if n else "\nall items available")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
