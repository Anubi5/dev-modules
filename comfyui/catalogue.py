"""Read-only local model inventory shared by humans and Hermes. No network or model loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
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
    from fetch_models import declarations
    for row in declarations(HERE/'models.txt'):
        if row['sha256']:pins['comfy/'+row['folder']+'/'+row['name']]=row['sha256'].lower()
    for package in registry.packages().values():
        for part in package['parts'].values(): pins['comfy/'+part['folder']+'/'+part['file']]=part['sha256']
    helpers=HERE/'helper_models.json'
    if helpers.exists():
        for row in json.loads(helpers.read_text())['files']: pins['comfy/'+row['path']]=row['sha256']
    rig=declared_by('ai-rig-models')                 # the rig module's own declaration, through its command
    for row in (rig or {}).get('weights',{}).get('files',[]): pins['rig/'+row['path']]=row['sha256']
    return pins


def declared_by(command,*arguments):
    """Another module's model declarations through its read-only command (JSON); None when it is not installed or
    does not answer. Never reads another module's files."""
    executable=shutil.which(command)
    if not executable: return None
    try:
        result=subprocess.run([executable,*arguments],capture_output=True,text=True,timeout=60)
        return json.loads(result.stdout) if result.returncode==0 else None
    except (OSError,ValueError,subprocess.TimeoutExpired): return None


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
        if registry.moved(p):continue        # profiles moved into families (3.2) are listed with the model profiles below
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
    for section in ('models','video_models'):              # {"family": NAME}: that family's profile, in place
        entries=legacy.get(section,{})
        if any(e=={'family':n} for n,e in entries.items()):
            from generation.models import PROFILES
            legacy[section]={n:(PROFILES[section][n] if e=={'family':n} else e) for n,e in entries.items()}
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
    chat=declared_by('ai-models','declared','--json') or {}   # the llama-swap module's own declaration
    declared_specs={row['id']:row for row in chat.get('chat',[])}
    if chat.get('resolved'):
        specs=declared_specs
        for name,p in chat['resolved'].items():
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
    family_metadata={n:p for n,p in registry.packages().items() if not registry.moved(p)}
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
    hashes,accepted=hash_record(),accepted_records()
    for package in packages:
        package.update(stage(root,package,by_path,hashes,accepted.get(package['id'])))
    return dict(schema='local-ai.models/1',root=str(root),packages=packages,files=files,discovery_read_only=True)


# Status shown to people and agents (Plan 3.6/3.8): declared -> installed -> verified -> accepted.
HASHES=Path.home()/'.local/share/modules/comfyui/models/.hashes'          # written by install/update verification
ACCEPTED=Path.home()/'.local/state/modules/comfyui/accepted.json'          # written by "accept" only
STAGES=('declared','installed','verified','accepted')


def hash_record():
    """{path: (sha256, size, mtime_ns)} of the files the module's install/update verification hashed (read-only)."""
    from fetch_models import load_record
    try: return load_record(str(HASHES))
    except (OSError,ValueError): return {}


def accepted_records():
    try:
        data=json.loads(ACCEPTED.read_text())
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError): return {}


def component_hashes(root,package,by_path,hashes):
    """{component: sha256} when every component's current size and time match a hash the verification recorded and
    that hash is the declared pin (where one exists); None otherwise. Reads no file contents."""
    found={}
    for component in package['components']:
        path=root/component; row=by_path.get(str(path.resolve()))
        try: st=path.stat()
        except OSError: return None
        known=hashes.get(str(path)) or hashes.get('/models/'+component)
        if not known or known[1]!=st.st_size or known[2]!=st.st_mtime_ns: return None
        if row and row.get('expected_sha256') and known[0]!=row['expected_sha256']: return None
        found[component]=known[0]
    return found


def stage(root,package,by_path,hashes,record):
    """One summary per model: declared (files absent), installed (present), verified (hashes match the pins),
    accepted (real output checked on this PC; date and commit recorded, for exactly these files)."""
    if package['missing'] or not package['components']: return dict(stage='declared',accepted=None)
    current=component_hashes(root,package,by_path,hashes)
    if current is None: return dict(stage='installed',accepted=None)
    if record and record.get('components')==current: return dict(stage='accepted',accepted=record)
    return dict(stage='verified',accepted=None)


def accept(report,query,record_path,note=None):
    """Record that a real run of a verified model was checked on this PC: its job record (status succeeded, this
    model, outputs present), the date, the module revision and the exact component hashes. @return the record"""
    selected=select(report,query)
    if selected not in report['packages']: raise ValueError('accept takes a model ID (ai models list)')
    if selected['stage'] not in ('verified','accepted'):
        raise ValueError(f"{query} is {selected['stage']}: only a verified model can be accepted (dev update verifies the files)")
    run=json.loads(Path(record_path).read_text())
    if run.get('status')!='succeeded' or run.get('model')!=selected['id']:
        raise ValueError(f"{record_path} is not a succeeded run of {selected['id']}")
    outputs=[o for o in run.get('outputs',[]) if Path(o).is_file()]
    if not outputs: raise ValueError(f'{record_path}: its outputs are missing')
    from datetime import datetime,timezone
    entry=dict(at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),revision=run.get('module'),
               record=str(Path(record_path).resolve()),outputs=outputs,note=note,
               components=component_hashes(Path(report['root']),selected,
                                           {str(Path(r['path']).resolve()):r for r in report['files']},hash_record()))
    data=accepted_records(); data[selected['id']]=entry
    ACCEPTED.parent.mkdir(parents=True,exist_ok=True)
    tmp=ACCEPTED.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=1,sort_keys=True)+'\n'); os.replace(tmp,ACCEPTED)
    return dict(schema='local-ai.model-acceptance/1',id=selected['id'],status='accepted',**entry)


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
    parser.add_argument('command',choices=['list','show','verify','accept'])
    parser.add_argument('query',nargs='?')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--all',action='store_true',help='list component files as well as logical packages')
    parser.add_argument('--details',action='store_true',help='list: the properties behind each status')
    parser.add_argument('--record',help='accept: the job record (.json) of a real run of this model, checked by you')
    parser.add_argument('--note',help='accept: what was checked')
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('MODELS_MOUNT','/models')))
    args=parser.parse_args(argv)
    if args.command=='list' and args.query: parser.error('list does not take a model ID; use show ID')
    if args.all and args.command!='list': parser.error('--all is only supported by list')
    if args.command!='list' and not args.query: parser.error('show/verify/accept require a model or package ID')
    if args.details and args.command!='list': parser.error('--details is only supported by list')
    if (args.record is None)!=(args.command!='accept'): parser.error('accept needs --record FILE (and only accept takes it)')
    try:
        report=inventory(args.root)
        result=(report if args.command=='list' else select(report,args.query) if args.command=='show'
                else accept(report,args.query,args.record,args.note) if args.command=='accept' else verify(args.root,report,args.query))
        if args.json or args.command!='list': print(json.dumps(result,indent=2))
        else:
            for row in report['packages']:
                print(f"{row.get('stage',row['status']):<10} {row['id']}")
                if args.details and 'stage' in row:
                    print(f"           files {row['status']}" + (f", missing {len(row['missing'])}" if row['missing'] else '')
                          + f"; {row['compatibility']}" + (f"; accepted {row['accepted']['at']} ({row['accepted'].get('revision')})" if row['accepted'] else ''))
            components={str((args.root/component).resolve()) for package in report['packages'] for component in package['components']}
            # Unrecognized files and standalone GGUF/LoRA files stay visible. A missing
            # registry reference is never evidence that a file is unused or disposable.
            visible=[row for row in report['files'] if args.all or row.get('kind')=='download-staging' or row['path'] not in components]
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
