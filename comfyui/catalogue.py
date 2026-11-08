"""Read-only local model inventory shared by humans and Hermes. No network or model loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from generation import registry

EXTENSIONS = {'.safetensors','.gguf','.ckpt','.pt','.pth','.bin','.onnx'}
HERE = Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def declared_pins():
    pins={}
    for line in (HERE/'models.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        fields=line.split(); name=next((x for x in fields[3:] if x!='optional' and not x.startswith('sha256:')),Path(fields[1]).name)
        pin=next((x[7:].lower() for x in fields[3:] if x.startswith('sha256:')),None)
        if pin: pins['comfy/'+fields[2]+'/'+name]=pin
    for package in registry.packages().values():
        for part in package['parts'].values(): pins['comfy/'+part['folder']+'/'+part['file']]=part['sha256']
    helpers=HERE/'helper_models.json'
    if helpers.exists():
        for row in json.loads(helpers.read_text())['files']: pins['comfy/'+row['path']]=row['sha256']
    rig=HERE.parent/'rig/models.json'
    if not rig.exists(): rig=Path.home()/'.local/share/modules/rig/models.json'
    if rig.exists():
        for row in json.loads(rig.read_text())['weights']['files']: pins['rig/'+row['path']]=row['sha256']
    return pins


def inventory(root):
    root=Path(root).resolve(); pins=declared_pins(); files=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
        if not path.resolve().is_relative_to(root): continue
        relative=path.relative_to(root).as_posix(); stat=path.stat()
        kind='lora' if 'loras' in path.parts else 'gguf' if path.suffix.lower()=='.gguf' else 'weights'
        status='untested' if relative in pins or kind in ('lora','gguf') else 'needs-something'
        files.append(dict(id=relative,kind=kind,path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                          status=status,expected_sha256=pins.get(relative),
                          reason='No GPU acceptance inferred from file discovery' if status=='untested' else 'No reviewed package or declaration recognizes this file'))
    packages=[]
    for name,p in registry.packages().items():
        missing=[]
        for role,part in p['parts'].items():
            path=root/'comfy'/part['folder']/part['file']
            if not path.is_file() or not path.resolve().is_relative_to(root) or path.stat().st_size!=part['bytes']:
                missing.append(dict(role=role,path=str(path),expected_bytes=part['bytes']))
        packages.append(dict(id=name,kind=p['kind'],architecture=p['architecture'],status='needs-something' if missing else 'ready',
                             missing=missing,integrity='use verify for full SHA-256 checking',experimental=p.get('experimental',True),
                             licence=p.get('licence',{}),package_sha256=p['sha256'],
                             components=['comfy/'+part['folder']+'/'+part['file'] for part in p['parts'].values()]))
    legacy=json.loads((HERE/'profiles.json').read_text())
    for name,p in dict(legacy['models'],**legacy.get('video_models',{})).items():
        references={p[key] for key in ['file','text_encoder','vae'] if isinstance(p.get(key),str)}
        if p.get('kind')=='3d':
            from generation.graph import apply_recipe
            recipe_path=(HERE/p['workflow']).resolve()
            if not recipe_path.is_relative_to(HERE): raise ValueError('Unsafe legacy recipe path')
            recipe=json.loads(recipe_path.read_text())
            graph=apply_recipe(recipe,dict(p.get('set',{}),prefix='inventory',
                **{key:'input.png' for key,spec in recipe['inputs'].items() if spec.get('upload')}))
            references.update(value for node in graph.values() for value in node['inputs'].values()
                              if isinstance(value,str) and Path(value).suffix.lower() in EXTENSIONS)
        components=[];missing=[]
        for reference in sorted(references):
            matches=[key for key in pins if key.startswith('comfy/') and (key.endswith('/'+reference) or Path(key).name==reference)]
            if len(matches)!=1:
                missing.append(reference+' (no unique component declaration)');continue
            key=matches[0];components.append(key);path=root/key
            if not path.is_file() or not path.resolve().is_relative_to(root): missing.append(key)
        packages.append(dict(id=name,kind=p.get('kind','image'),architecture=p.get('arch'),
            status='needs-something' if missing else 'ready',missing=missing,components=components,
            experimental=True,integrity='use verify for full SHA-256 checking'))
    rig=[key for key in pins if key.startswith('rig/')]
    if rig:
        missing=[key for key in rig if not (root/key).is_file() or not (root/key).resolve().is_relative_to(root)]
        packages.append(dict(id='unirig',kind='rig',status='needs-something' if missing else 'ready',missing=missing,
                             components=rig,experimental=True))
    # Installed local chat selections are metadata; never invoke a server or discover executable code.
    resolved=Path.home()/'.local/share/modules/llama-swap/models/models.json'
    if resolved.exists():
        for name,p in json.loads(resolved.read_text()).items():
            paths=p.get('files') or ([p['file']] if p.get('file') else [])
            if not paths: continue
            missing=[str(path) for path in map(Path,paths) if not path.is_file() or not path.resolve().is_relative_to(root)]
            packages.append(dict(id=name,kind='chat',status='needs-something' if missing else 'ready',missing=missing,
                                 components=paths,experimental=True))
    return dict(schema='local-ai.models/1',root=str(root),packages=packages,files=files,discovery_read_only=True)


def select(report, query):
    exact=[x for x in report['packages']+report['files'] if x['id']==query or x.get('path')==query]
    matches=exact or [x for x in report['files'] if Path(x['id']).name==query]
    if len(matches)!=1: raise ValueError('Expected one exact model/package match; found '+str(len(matches)))
    return matches[0]


def verify(root, report, query):
    selected=select(report,query); wanted=[]
    if selected in report['files']: wanted=[selected]
    else:
        if selected['missing']: raise ValueError('Package has missing or changed-size components')
        for component in selected['components']: wanted.append(select(report,component))
    verified=[]
    for row in wanted:
        path=Path(row['path']); actual=sha256(path); expected=row.get('expected_sha256')
        if expected and actual!=expected: raise ValueError('SHA-256 mismatch; file preserved: '+row['id'])
        verified.append(dict(id=row['id'],sha256=actual,pinned=bool(expected),bytes=path.stat().st_size))
    return dict(schema='local-ai.model-verification/1',id=selected['id'],status='verified',files=verified,
                meaning='Content matches declared pins where available; this is not GPU or quality acceptance')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['list','show','verify'])
    parser.add_argument('query',nargs='?')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('MODELS_MOUNT','/models')))
    args=parser.parse_args(argv)
    if args.command!='list' and not args.query: parser.error('show/verify require a model or package ID')
    try:
        report=inventory(args.root)
        result=report if args.command=='list' else select(report,args.query) if args.command=='show' else verify(args.root,report,args.query)
        if args.json or args.command!='list': print(json.dumps(result,indent=2))
        else:
            for row in report['packages']+report['files']: print(f"{row['status']:<16} {row['id']}")
        return 0
    except (OSError,ValueError,KeyError) as error:
        if args.json: print(json.dumps(dict(status='failed',error=str(error))))
        else: print('CHECK '+str(error),file=sys.stderr)
        return 1


if __name__=='__main__': raise SystemExit(main())
