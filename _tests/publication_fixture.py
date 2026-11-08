"""Disposable publication checks using the actual fetch_url implementation and fake HTTP bytes.

The PC entry point requires a Windows-created nonce on an existing writable
Windows-backed mount. It never mounts a drive, downloads, or accesses model data.
Failed staging is deliberately retained, matching the accepted download policy.
"""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import urllib.request
from unittest.mock import patch

CASES=('successful_publication','destination_exists_refusal','racing_destination_refusal',
       'checksum_failure','interrupted_transfer')


def exercise(root,source):
    root=Path(root)
    module=ModuleType('reviewed_fixture_fetch')
    hub=SimpleNamespace(HfApi=lambda **kw:None,hf_hub_download=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
    with patch.dict(sys.modules,huggingface_hub=hub):
        exec(compile(source,'reviewed-fetch_models.py','exec'),module.__dict__)
    data=b'EVA disposable publication fixture\n';sha=hashlib.sha256(data).hexdigest()
    results={name:dict(status='SKIPPED') for name in CASES}

    class Response(io.BytesIO):
        headers={'Content-Length':str(len(data))}

    class Interrupted(Response):
        def read(self,size=-1):
            if self.tell():raise OSError('Simulated interrupted fixture transfer')
            return super().read(7)

    for name in CASES:
        folder=root/name;folder.mkdir(exist_ok=False);destination=folder/'published.fixture'
        owner=b'Existing owner fixture; preserve verbatim\n'
        try:
            if name=='destination_exists_refusal':destination.write_bytes(owner)
            real_link=os.link
            def racing_link(src,dst):
                with open(dst,'xb') as stream:stream.write(owner)
                return real_link(src,dst)
            response=Interrupted(data) if name=='interrupted_transfer' else Response(data)
            with patch.object(urllib.request,'urlopen',return_value=response), \
                 patch('socket.create_connection',side_effect=AssertionError('Network forbidden')), \
                 patch.object(module.os,'link',side_effect=racing_link if name=='racing_destination_refusal' else real_link):
                error=None
                try:module.fetch_url('https://fixture.invalid/no-network',str(destination),'0'*64 if name=='checksum_failure' else sha)
                except (OSError,RuntimeError) as caught:error=caught
            staging=list(folder.glob('*.part'))
            if len(staging)!=1:raise AssertionError('Expected exactly one retained staging fixture')
            if name=='successful_publication':
                if error or destination.read_bytes()!=data or not os.path.samefile(staging[0],destination):
                    raise AssertionError('Publication did not create a matching hard link')
            elif name in ('destination_exists_refusal','racing_destination_refusal'):
                if error is None or destination.read_bytes()!=owner:raise AssertionError('Existing destination not preserved')
            elif error is None or destination.exists():raise AssertionError('Failed transfer published a destination')
            results[name]=dict(status='PASS',staging='retained',destination='published hard link' if name=='successful_publication' else 'preserved' if destination.exists() else 'absent')
        except KeyboardInterrupt:
            results[name]=dict(status='CANCELLED',staging='retained');break
        except Exception as error:results[name]=dict(status='FAIL',error=type(error).__name__+': '+str(error))
    status='CANCELLED' if any(row['status']=='CANCELLED' for row in results.values()) else 'PASS' if all(row['status']=='PASS' for row in results.values()) else 'FAIL'
    return dict(status=status,cases=results,
                cleanup='No files deleted; successful and failed staging retained for owner review',network='fake response only')


def main():
    root=Path(sys.argv[1]);nonce=sys.argv[2]
    result=dict(status='SKIPPED',cases={name:dict(status='SKIPPED') for name in CASES})
    try:
        resolved=root.resolve(strict=True)
        if root.is_symlink() or any(resolved.is_relative_to(path) for path in (Path('/models'),Path('/models-rw'))):
            raise ValueError('Refusing model-store paths or symlink fixture root')
        if (root/'windows-nonce.txt').read_text()!=nonce:raise ValueError('Windows/WSL fixture mapping is not confirmed')
        if os.statvfs(root).f_flag & os.ST_RDONLY:raise ValueError('Fixture mount is read-only; nothing changed')
        mounts=[]
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            fields=line.split();cut=fields.index('-');point=Path(fields[4].replace('\\040',' '))
            if resolved.is_relative_to(point):mounts.append((len(point.parts),fields[cut+1]))
        filesystem=max(mounts)[1]
        if filesystem not in ('9p','drvfs'):raise ValueError('An existing Windows-backed fixture mount is required')
        source=gzip.decompress(base64.b64decode(sys.argv[3])).decode('utf-8')
        result=exercise(root,source);result.update(filesystem=filesystem,source_sha256=hashlib.sha256(source.encode()).hexdigest())
    except KeyboardInterrupt:result.update(status='CANCELLED')
    except (OSError,ValueError) as error:result.update(reason=str(error))
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 3 if result['status']=='SKIPPED' else 130 if result['status']=='CANCELLED' else 1


if __name__=='__main__':raise SystemExit(main())
