"""Install-time asset verification. Runtime inference never downloads models."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    module = Path(os.environ["MOD"])
    manifest = json.loads((module / "models.json").read_text())
    root = Path(os.environ.get("RIG_MODELS_ROOT", str(Path(os.environ.get("MODELS_MOUNT", "/models")) / "rig")))
    write_root = os.environ.get("RIG_MODELS_WRITE_ROOT")
    if not write_root and os.environ.get("MODELS_RW"):
        write_root = str(Path(os.environ["MODELS_RW"]) / "rig")
    installed = {"source_revision": manifest["source"]["revision"], "models_root": str(root), "checkpoints": []}
    for row in manifest["weights"]["files"]:
        target = root / row["path"]
        if not target.is_file():
            if not write_root:
                raise SystemExit("Missing pinned model: " + str(target) + "; use the Windows module-update command with the writable model mount")
            from huggingface_hub import hf_hub_download
            downloaded = Path(hf_hub_download(repo_id=manifest["weights"]["repository"],
                revision=manifest["weights"]["revision"], filename=row["path"], token=False))
            if sha256(downloaded) != row["sha256"]:
                raise SystemExit("Downloaded checkpoint checksum mismatch: " + row["path"])
            writable = Path(write_root) / row["path"]
            writable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, writable)
        if not target.is_file() or sha256(target) != row["sha256"]:
            raise SystemExit("Pinned model missing or checksum mismatch: " + row["path"])
        installed["checkpoints"].append({"path": str(target), "sha256": row["sha256"]})
        print("Verified", row["path"], flush=True)
    architecture = json.loads((module / "architecture.json").read_text())
    destination = module / "app/architecture/opt350m/config.json"
    if not destination.is_file() or sha256(destination) != architecture["sha256"]:
        cache = Path(os.environ.get("RIG_ASSET_CACHE", str(Path.home() / ".cache/huggingface/hub")))
        cached = cache / "models--facebook--opt-350m/snapshots" / architecture["revision"] / architecture["path"]
        if not cached.is_file():
            from huggingface_hub import hf_hub_download
            cached = Path(hf_hub_download(repo_id=architecture["repository"], revision=architecture["revision"],
                                         filename=architecture["path"], token=False))
        if sha256(cached) != architecture["sha256"]:
            raise SystemExit("Architecture configuration checksum mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
    installed["architecture"] = {"directory": str(destination.parent), "sha256": architecture["sha256"]}
    (module / "app/installed.json").write_text(json.dumps(installed, indent=2))


if __name__ == "__main__":
    main()
