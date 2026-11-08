"""Read-only local model inventory shared by humans and Hermes. No network or model loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from generation import registry, builders, video

EXTENSIONS = {'.safetensors','.gguf','.ckpt','.pt','.pth','.bin','.onnx'}
HERE = Path(__file__).resolve().parent
CAPABILITIES = {'llm','vlm','image','video','3d','upscaling'}


def format_info(path):
    """Bounded, non-executing format inspection. Never unpickle a checkpoint.

    Headers describe a container, not a supported model architecture or ability.
    Full integrity remains the explicit verify operation.
    """
    size=path.stat().st_size; suffix=path.suffix.lower()
    result=dict(name=suffix.lstrip('.'),status='unverified',completeness='unknown')
    try:
        with path.open('rb') as stream:
            if suffix=='.gguf':
                header=stream.read(24)
                if len(header)!=24 or header[:4]!=b'GGUF' or int.from_bytes(header[4:8],'little') not in (2,3):
                    raise ValueError('Invalid or unsupported GGUF header')
                result.update(status='recognized',completeness='header-only',version=int.from_bytes(header[4:8],'little'))
            elif suffix=='.safetensors':
                prefix=stream.read(8); length=int.from_bytes(prefix,'little')
                if len(prefix)!=8 or not 2<=length<=min(16*1024**2,size-8):
                    raise ValueError('Invalid or oversized safetensors header')
                metadata=json.loads(stream.read(length))
                if not isinstance(metadata,dict): raise ValueError('Invalid tensor metadata')
                spans=[]
                for name,tensor in metadata.items():
                    if name=='__metadata__': continue
                    start,end=tensor['data_offsets']
                    if type(start) is not int or type(end) is not int or not 0<=start<=end<=size-8-length:
                        raise ValueError('Tensor data is truncated or offsets invalid')
                    spans.append((start,end))
                spans.sort()
                if not spans or any(left[1]>right[0] for left,right in zip(spans,spans[1:])):
                    raise ValueError('Empty or overlapping tensor data')
                result.update(status='recognized',completeness='declared-ranges-present')
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        result.update(status='invalid',completeness='invalid',reason=str(error))
    return result


def tags(metadata):
    values=metadata.get('capability_tags',[])
    if not isinstance(values,list) or any(value not in CAPABILITIES for value in values):
        raise ValueError('Invalid owning-package capability tags')
    return sorted(set(values))


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
        if not path.is_file(): continue
        if not path.resolve().is_relative_to(root): continue
        staged=(any(part.startswith('.model-stage-') for part in path.relative_to(root).parts)
                or bool(re.search(r'\.[0-9a-f]{32}\.part$',path.name)))
        if staged and '.cache' in path.parts and not path.name.endswith('.incomplete'):continue
        if staged:
            stat=path.stat()
            files.append(dict(id=path.relative_to(root).as_posix(),kind='download-staging',path=str(path),
                bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,status='failed-download',expected_sha256=None,
                reason='Failed or incomplete download retained for owner review; never executed or automatically deleted'))
            continue
        if path.suffix.lower() not in EXTENSIONS: continue
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
                             components=['comfy/'+part['folder']+'/'+part['file'] for part in p['parts'].values()],
                             description=p.get('description',name),defaults=p.get('defaults',{}),capabilities=p.get('capabilities',{})))
    legacy=json.loads((HERE/'profiles.json').read_text())
    for name,p in dict(legacy['models'],**legacy.get('video_models',{}),**legacy.get('assets',{})).items():
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
            experimental=True,integrity='use verify for full SHA-256 checking',description=p.get('about',name),
            defaults={key:p[key] for key in ['steps','cfg','sampler','scheduler','style','fps'] if key in p}))
    rig=[key for key in pins if key.startswith('rig/')]
    if rig:
        missing=[key for key in rig if not (root/key).is_file() or not (root/key).resolve().is_relative_to(root)]
        packages.append(dict(id='unirig',kind='rig',status='needs-something' if missing else 'ready',missing=missing,
                             components=rig,experimental=True))
    # Installed local chat selections are metadata; never invoke a server or discover executable code.
    declared_chat=HERE.parent/'llama-swap/models.json'
    declared_specs={row['id']:row for row in json.loads(declared_chat.read_text())['chat']} if declared_chat.exists() else {}
    resolved=Path.home()/'.local/share/modules/llama-swap/models/models.json'
    if resolved.exists():
        source=resolved.parents[1]/'models.json'
        specs={p['id']:p for p in json.loads(source.read_text())['chat']} if source.exists() else {}
        for name,p in json.loads(resolved.read_text()).items():
            paths=p.get('files') or ([p['file']] if p.get('file') else [])
            if not paths: continue
            missing=[str(path) for path in map(Path,paths) if not path.is_file() or not path.resolve().is_relative_to(root)]
            spec=specs.get(name,{})
            if not spec:missing.append('No current reviewed serving entry')
            effective=dict(specs.get(spec.get('weights_from'),{}),**spec)
            context=re.search(r'(?:^|\s)(?:-c|--ctx-size)\s+(\d+)',effective.get('args',''))
            packages.append(dict(id=name,kind='chat',status='needs-something' if missing else 'ready',missing=missing,
                                 components=paths,experimental=True,description=effective.get('name',name),
                                 context_tokens=int(context[1]) if context else None,weights_from=spec.get('weights_from')))
    # Enrich the existing view additively; no second inventory or serving registry.
    family_metadata=registry.packages()
    legacy_metadata=dict(legacy['models'],**legacy.get('video_models',{}),**legacy.get('assets',{}))
    by_path={str(Path(row['path']).resolve()):row for row in files}
    for row in files:
        if row['kind']=='download-staging':
            row.update(format=dict(name=Path(row['path']).suffix.lstrip('.'),status='unverified',completeness='failed-download'),
                capability_tags=[],packages=[],runnable=False,identification='failed-download',compatibility='not runnable; owner cleanup only')
            continue
        row.update(format=format_info(Path(row['path'])),capability_tags=[],packages=[],runnable=False,
                   identification='declared-pin' if row['expected_sha256'] else 'unknown',
                   compatibility='no reviewed serving package')
    for package in packages:
        name=package['id']
        metadata=family_metadata.get(name,legacy_metadata.get(name,{}))
        if package['kind']=='chat':
            spec=declared_specs.get(name,specs.get(name,{}))
            metadata=dict(declared_specs.get(spec.get('weights_from'),specs.get(spec.get('weights_from'),{})),**spec)
        if package['kind']=='rig': metadata={}
        package['capability_tags']=tags(metadata)
        package['formats']=[]; invalid=[]
        for component in package['components']:
            row=by_path.get(str((root/component).resolve()))
            if not row: continue
            if row['kind']=='download-staging':invalid.append(row['id']);continue
            row['packages'].append(name)
            row['identification']='declared-package'
            row['capability_tags']=sorted(set(row['capability_tags']+package['capability_tags']))
            package['formats'].append(row['format']['name'])
            if row['format']['status']=='invalid': invalid.append(row['id'])
        package['formats']=sorted(set(package['formats']))
        supported=(name in family_metadata or package['kind'] in ('chat','rig','upscaling')
                   or (package['kind']=='image' and metadata.get('arch') in builders.BUILTINS)
                   or (package['kind']=='video' and metadata.get('arch') in video.ARCHITECTURES)
                   or (package['kind']=='3d' and bool(metadata.get('workflow'))))
        package['completeness']='missing-components' if package['missing'] else 'invalid-format' if invalid else 'declared-components-present'
        package['compatibility']='invalid component format' if invalid else 'requires a reviewed family adapter' if not supported else 'reviewed package; hardware acceptance separate'
        package['runnable']=package['status']=='ready' and supported and not invalid and package['kind']!='upscaling'
        for component in package['components']:
            row=by_path.get(str((root/component).resolve()))
            if row:
                row['compatibility']='component of '+', '.join(row['packages'])
                # A component is not an independently runnable model.
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
    parser.add_argument('--all',action='store_true',help='list component files as well as logical packages')
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('MODELS_MOUNT','/models')))
    args=parser.parse_args(argv)
    if args.command=='list' and args.query: parser.error('list does not take a model ID; use show ID')
    if args.all and args.command!='list': parser.error('--all is only supported by list')
    if args.command!='list' and not args.query: parser.error('show/verify require a model or package ID')
    try:
        report=inventory(args.root)
        result=report if args.command=='list' else select(report,args.query) if args.command=='show' else verify(args.root,report,args.query)
        if args.json or args.command!='list': print(json.dumps(result,indent=2))
        else:
            for row in report['packages']: print(f"{row['status']:<16} {row['id']}")
            components={str((args.root/component).resolve()) for package in report['packages'] for component in package['components']}
            # Unrecognized files and standalone GGUF/LoRA files stay visible. A missing
            # registry reference is never evidence that a file is unused or disposable.
            visible=[row for row in report['files'] if args.all or row['path'] not in components]
            for row in visible:
                if row.get('kind')=='download-staging':print(f"failed download  {row['bytes']} bytes  {row['path']}")
                else:print(f"{row['status']:<16} {row['id']}")
            hidden=len(report['files'])-len(visible)
            if hidden: print(f"{hidden} declared component file(s) hidden; use --all for details")
        return 0
    except (OSError,ValueError,KeyError) as error:
        if args.json: print(json.dumps(dict(status='failed',error=str(error))))
        else: print('CHECK '+str(error),file=sys.stderr)
        return 1


if __name__=='__main__': raise SystemExit(main())
