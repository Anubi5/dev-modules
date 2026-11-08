"""Verify manually supplied helper models; never download or replace model data."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or '\\' in relative or ':' in relative or Path(relative).is_absolute():
        raise ValueError('Expected a portable model path')
    target = (root/relative).resolve()
    if not target.is_relative_to(root): raise ValueError('Model path escapes its root')
    return target


def check_workflow(root, workflow, manifest=None):
    """Cheap capability preflight: no hashing, service changes or downloads."""
    classes = {node.get('class_type', '') for node in (workflow or {}).values()}
    prefixes = []
    if 'DWPreprocessor' in classes: prefixes.append(('pose', 'controlnet_aux/'))
    if classes & {'IPAdapterUnifiedLoaderFaceID', 'IPAdapterFaceID', 'InsightFaceLoader'}:
        prefixes.append(('FaceID', 'insightface/'))
    if not prefixes: return
    manifest = manifest or json.loads(Path(__file__).with_suffix('.json').read_text())
    for feature, prefix in prefixes:
        for row in manifest['files']:
            if not row['path'].startswith(prefix): continue
            path = inside(root, row['path'])
            if not path.is_file() or path.stat().st_size != row['bytes']:
                raise ValueError(f"{feature} needs {row['path']} on D:\\AI\\models\\comfy "
                                 "with the declared size; copy it manually and run helper_models.py --root /models/comfy")


def verify(root, manifest=None, retire_from=None, allow_missing=False):
    manifest = manifest or json.loads(Path(__file__).with_suffix('.json').read_text())
    if manifest.get('schema') != 'local-ai.helper-models/1' or not manifest.get('files'):
        raise ValueError('Unsupported helper model manifest')
    retired, missing = [], []
    for row in manifest['files']:
        path = inside(root, row['path'])
        if allow_missing and not path.exists():
            missing.append(row['path'])
            continue
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Missing or changed helper: '+str(path)+'; copy the declared file to D:\\AI\\models\\comfy manually')
        if retire_from is not None:
            old = Path(retire_from)/row['legacy']
            if old.is_symlink(): continue
            if old.exists():
                old = inside(retire_from, row['legacy'])
                if old.is_relative_to(Path(root).resolve()):
                    raise ValueError('Refusing to retire anything in the permanent model store')
                if not old.is_file() or old.stat().st_size != row['bytes'] or digest(old) != row['sha256']:
                    raise ValueError('Legacy helper differs; both files preserved: '+str(old))
                retired.append(old)
    # Only the explicitly requested future module update retires byte-identical
    # duplicates. Read-only checks and the current staged PC test never do so.
    for old in retired: old.unlink()
    result = dict(status='verified', files=len(manifest['files'])-len(missing),
                  retired_duplicates=[str(p) for p in retired])
    if allow_missing: result['missing_optional'] = missing
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--retire-from', type=Path)
    parser.add_argument('--allow-missing', action='store_true', help='Install/update: report absent optional helpers')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, retire_from=args.retire_from, allow_missing=args.allow_missing)))
        return 0
    except (OSError, ValueError) as error:
        print('CHECK '+str(error))
        return 1


if __name__ == '__main__': raise SystemExit(main())
