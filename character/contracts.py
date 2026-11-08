"""Versioned, hashed file hand-offs. No model/runtime imports or execution."""
import hashlib
import json
from pathlib import Path


ASSET_SCHEMA = "local-ai.asset/1"
KINDS = {"image", "video", "mesh", "rig", "animation"}
EXTENSIONS = {"image": {".png", ".jpg", ".jpeg", ".webp"}, "video": {".webp", ".mp4", ".webm"},
              "mesh": {".glb", ".fbx", ".obj"}, "rig": {".fbx", ".glb", ".blend"},
              "animation": {".fbx", ".glb"}}


def digest(file):
    result = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or Path(relative).is_absolute():
        raise ValueError("Expected a relative artifact path")
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path leaves its workspace: " + relative)
    return path


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+".pending")
    temporary.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def validate(data, root, check_files=True):
    if not isinstance(data, dict) or data.get("schema") != ASSET_SCHEMA or not isinstance(data.get("kind"), str) or data["kind"] not in KINDS:
        raise ValueError("Unsupported artifact schema or kind")
    validation = data.get("validation", {})
    if not isinstance(validation, dict) or validation.get("numerical") not in ("passed", "failed", "not_run"):
        raise ValueError("Missing numerical validation state")
    if validation.get("visual") not in ("passed", "failed", "not_established"):
        raise ValueError("Missing visual validation state")
    if validation.get("generalization") != "not_established":
        raise ValueError("Generalization is not established by this contract")
    coordinates = data.get("coordinates")
    if data["kind"] in {"mesh", "rig", "animation"}:
        if not isinstance(coordinates, dict) or coordinates.get("units") not in ("m", "cm") or \
                coordinates.get("up") != "+Z" or coordinates.get("forward") not in ("-Y", "+X"):
            raise ValueError("Geometry requires explicit consumption coordinates")
    if data["kind"] == "rig" and data.get("experimental") is not True:
        raise ValueError("Rig artifacts must retain experimental status")
    if data.get("source") is not None and not isinstance(data["source"], dict):
        raise ValueError("Artifact provenance must be an object")
    source = data.get("source") or {}
    if "record" in source:
        record = inside(root, source["record"])
        if check_files and (not record.is_file() or digest(record) != source.get("sha256")):
            raise ValueError("Missing or changed producer record")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Artifact has no files")
    roles = set()
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str) or not row["role"] or row["role"] in roles:
            raise ValueError("Artifact file roles must be unique")
        roles.add(row["role"])
        file = inside(root, row.get("path"))
        if file.suffix.lower() not in EXTENSIONS[data["kind"]]:
            raise ValueError("Unexpected artifact file type: " + file.name)
        checksum = row.get("sha256", "")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid artifact checksum")
        if isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
            raise ValueError("Invalid artifact byte count")
        if check_files and (not file.is_file() or file.stat().st_size != row.get("bytes") or digest(file) != checksum):
            raise ValueError("Missing or changed artifact: " + row["path"])
    return data


def read(path, check_files=True):
    path = Path(path)
    return validate(json.loads(path.read_text(encoding="utf-8")), path.parent, check_files)


def write(path, kind, files, coordinates=None, numerical="not_run", source=None):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("Refusing to replace artifact manifest: " + str(path))
    rows = []
    for role, file in files:
        file = Path(file).resolve()
        if kind == 'rig' and file.suffix.lower() != '.fbx':
            raise ValueError('New rig manifests are FBX-only; keep other formats in the producer record')
        relative = file.relative_to(path.parent).as_posix()
        rows.append(dict(role=role, path=relative, bytes=file.stat().st_size, sha256=digest(file)))
    data = dict(schema=ASSET_SCHEMA, kind=kind, files=rows, coordinates=coordinates,
                experimental=kind in {"rig", "animation"}, source=source,
                validation=dict(numerical=numerical, visual="not_established", generalization="not_established"))
    if kind == "rig": data["limitations"] = ["F6: OPEN — DEFERRED", "Automatic crash recovery is not established"]
    validate(data, path.parent)
    atomic_json(path, data)
    return data
