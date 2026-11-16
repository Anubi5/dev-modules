"""Two-phase PC fixture: real planner/fetcher, synthetic bytes, no real network."""
import base64
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

DATA=b'Owner-declined synthetic test file'
SHA=hashlib.sha256(DATA).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    mode=sys.argv[1]
    if mode=='create':
        bundle=json.loads(gzip.decompress(base64.b64decode(sys.argv[2])))
        root=Path(tempfile.mkdtemp(prefix='eva-windows-approval-'));root.chmod(0o700)
        module=root/'source/comfyui';module.mkdir(parents=True)
        (module/'fetch_models.py').write_text(bundle['fetch'])
        (root/'weights').mkdir();(root/'state').mkdir()
        repo='https://fixture.invalid/missing-test.safetensors'
        (module/'models.txt').write_text(repo+' missing-test.safetensors checkpoints sha256:'+SHA+'\n')
        (module/'download_licenses.json').write_text(json.dumps({repo:dict(approved=True,license='synthetic-test-only')}))
        family=module/'families/fixture';family.mkdir(parents=True)
        (family/'package.json').write_text(json.dumps(dict(parts=dict(model=dict(folder='checkpoints',file='missing-test.safetensors',bytes=len(DATA),sha256=SHA)))))
        print(json.dumps(dict(fixture=str(root))));return 0
    root=Path(sys.argv[2]).resolve()
    if not root.name.startswith('eva-windows-approval-') or root.parent!=Path(tempfile.gettempdir()).resolve():
        raise ValueError('Unexpected fixture root')
    if mode=='plan':
        print(json.dumps(load('approval_plan',root/'source/comfyui/fetch_models.py').download_plan(root/'source/comfyui/models.txt',root/'weights')));return 0
    if mode=='apply':
        approvals=sys.argv[3:]
        if any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in approvals):raise ValueError('Invalid hash argument')
        calls=[]
        def forbidden(*args,**kwargs):raise AssertionError('Real network forbidden')
        import socket
        import urllib.request
        socket.create_connection=forbidden
        class Response(io.BytesIO):
            headers={'Content-Length':str(len(DATA))}
        def fake_http(request,**kwargs):
            if request.full_url!='https://fixture.invalid/missing-test.safetensors':raise AssertionError('Unexpected source')
            calls.append(request.full_url);return Response(DATA)
        urllib.request.urlopen=fake_http
        sys.modules['huggingface_hub']=SimpleNamespace(HfApi=forbidden,hf_hub_download=forbidden)
        fetch=load('approval_fetch',root/'source/comfyui/fetch_models.py')
        os.environ.update(DEV_MODEL_OPERATION='approved' if approvals else 'verify',DEV_APPROVED_SHA256=json.dumps(approvals),DEV_UNATTENDED='')
        sys.argv=['fetch','--allow-downloads',str(root/'source/comfyui/models.txt'),str(root/'weights/comfy'),str(root/'weights/comfy'),str(root/'state')]
        # Writable fixture root is created only in the approved execution phase.
        if approvals:(root/'weights/comfy').mkdir(exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()) as output:code=fetch.main()
        target=root/'weights/comfy/checkpoints/missing-test.safetensors'
        print(json.dumps(dict(exit=code,approved_sha256=approvals,synthetic_transfers=len(calls),
            published_bytes=target.stat().st_size if target.exists() else 0,
            published_sha256=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None,
            staging_files=len(list((root/'weights').rglob('*.part'))),messages=output.getvalue())))
        return code
    if mode=='inspect':
        print(json.dumps(dict(weight_files=len([p for p in (root/'weights').rglob('*') if p.is_file()]))));return 0
    raise ValueError('Unexpected fixture mode')


if __name__=='__main__':raise SystemExit(main())
