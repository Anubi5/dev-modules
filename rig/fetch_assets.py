"""Install-time asset verification. Runtime inference never downloads models."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import sys
import tempfile


def declarations(module):
    return (json.loads((module/'models.json').read_text()),json.loads((module/'architecture.json').read_text()))


def download_plan(module=None, root=None):
    module=Path(module or os.environ.get('MOD',Path(__file__).resolve().parent))
    root=Path(root or os.environ.get('MODELS_MOUNT','/models'));manifest,architecture=declarations(module);files=[]
    weights=manifest['weights']
    entries=[(row,weights,root/'rig'/row['path'],'rig/'+row['path']) for row in weights['files']]
    entries.append((architecture,architecture,module/'app/architecture/opt350m/config.json','rig/architecture/'+architecture['path']))
    for row,owner,target,relative in entries:
        if Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or '\\' in row['path']:raise ValueError('Unsafe declared model path')
        if target.is_file():continue
        policy=owner.get('download_policy',{});sha=row.get('sha256');size=row.get('bytes')
        eligible=bool(re.fullmatch('[a-fA-F0-9]{64}',sha or '')) and bool(re.fullmatch('[a-fA-F0-9]{40}',owner.get('revision',''))) and type(size) is int and size>0 and policy.get('approved') is True and bool(policy.get('license')) and not policy.get('gated')
        files.append(dict(path=relative,bytes=size,source=owner['repository']+'@'+owner['revision']+'/'+row['path'],licence=policy.get('license'),sha256=sha.lower() if sha else None,eligible=eligible,reason=None if eligible else 'Manual copy: incomplete or restricted declaration'))
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


def acquire(repository,revision,row,destination,policy):
    if os.environ.get('DEV_MODEL_OPERATION')!='approved' or os.environ.get('DEV_UNATTENDED'):
        raise SystemExit('Missing pinned asset; run dev mod update rig for confirmed acquisition, or copy manually')
    if (policy.get('approved') is not True or not policy.get('license') or policy.get('gated')
            or not re.fullmatch('[a-fA-F0-9]{40}',revision)
            or not re.fullmatch('[a-fA-F0-9]{64}',row.get('sha256',''))
            or type(row.get('bytes')) is not int or row['bytes']<=0):
        raise SystemExit('Missing reviewed size/hash/licence; gated assets require manual owner acquisition')
    print(f"Missing pinned file: {row['path']}; bytes={row['bytes']}; source={repository}@{revision}; licence={policy['license']}; SHA-256={row['sha256']}",flush=True)
    if not approved_download(row['sha256']):
        raise SystemExit('SHA-256 not approved; download refused')
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=tempfile.mkdtemp(prefix='.model-stage-',dir=destination.parent)
    from huggingface_hub import hf_hub_download
    downloaded=Path(hf_hub_download(repo_id=repository,revision=revision,filename=row['path'],token=False,local_dir=staging))
    if sha256(downloaded)!=row['sha256'] or downloaded.stat().st_size!=row['bytes']:
        raise SystemExit('Downloaded file differs from declared pins; staging preserved')
    os.link(downloaded,destination) # preserve any existing destination, including races
    os.unlink(downloaded) # remove staging name only after verified publication


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if '--plan' in sys.argv or '--json' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('--plan',action='store_true',required=True);parser.add_argument('--json',action='store_true',required=True);parser.add_argument('--models-root');parser.add_argument('--module')
        args=parser.parse_args();print(json.dumps(download_plan(args.module,args.models_root)));return
    module = Path(os.environ["MOD"])
    manifest,architecture = declarations(module)
    root = Path(os.environ.get("RIG_MODELS_ROOT", str(Path(os.environ.get("MODELS_MOUNT", "/models")) / "rig")))
    write_root = os.environ.get("RIG_MODELS_WRITE_ROOT")
    if not write_root and os.environ.get("MODELS_RW"):
        write_root = str(Path(os.environ["MODELS_RW"]) / "rig")
    installed = {"source_revision": manifest["source"]["revision"], "models_root": str(root), "checkpoints": []}
    for row in manifest["weights"]["files"]:
        target = root / row["path"]
        if not target.resolve().is_relative_to(root.resolve()):raise SystemExit('Unsafe model path')
        if not target.is_file():
            if not write_root:
                raise SystemExit("Missing pinned model: " + str(target) + "; run dev mod update rig, or copy manually")
            writable = Path(write_root) / row["path"]
            acquire(manifest['weights']['repository'],manifest['weights']['revision'],row,writable,manifest['weights'].get('download_policy',{}))
        if not target.is_file() or sha256(target) != row["sha256"]:
            raise SystemExit("Pinned model missing or checksum mismatch: " + row["path"])
        installed["checkpoints"].append({"path": str(target), "sha256": row["sha256"]})
        print("Verified", row["path"], flush=True)
    destination = module / "app/architecture/opt350m/config.json"
    if not destination.is_file() or sha256(destination) != architecture["sha256"]:
        cache = Path(os.environ.get("RIG_ASSET_CACHE", str(Path.home() / ".cache/huggingface/hub")))
        cached = cache / "models--facebook--opt-350m/snapshots" / architecture["revision"] / architecture["path"]
        if not cached.is_file():
            acquire(architecture['repository'],architecture['revision'],architecture,destination,architecture.get('download_policy',{}))
            cached=destination
        if sha256(cached) != architecture["sha256"]:
            raise SystemExit("Architecture configuration checksum mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if cached!=destination:shutil.copyfile(cached, destination)
    installed["architecture"] = {"directory": str(destination.parent), "sha256": architecture["sha256"]}
    (module / "app/installed.json").write_text(json.dumps(installed, indent=2))


if __name__ == "__main__":
    main()
