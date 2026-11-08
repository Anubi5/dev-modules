"""Owner console check of the actual prompt; only synthetic pinned data, never network."""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main():
    source=gzip.decompress(base64.b64decode(sys.argv[1])).decode()
    root=Path(tempfile.mkdtemp(prefix='eva-decline-prompt-'));root.chmod(0o700)
    path=root/'fetch_models.py';path.write_text(source)
    data=b'Owner-declined synthetic test file';sha=hashlib.sha256(data).hexdigest()
    (root/'models.txt').write_text('fixture/repo missing-test.safetensors checkpoints sha256:'+sha+'\n')
    (root/'download_licenses.json').write_text(json.dumps({'fixture/repo':dict(approved=True,license='synthetic-test-only')}))
    weights=root/'weights';weights.mkdir();calls=[]
    def forbidden(*args,**kwargs):calls.append('download');raise AssertionError('Downloads forbidden in this fixture')
    info=SimpleNamespace(gated=False,siblings=[SimpleNamespace(rfilename='missing-test.safetensors',size=len(data))])
    sys.modules['huggingface_hub']=SimpleNamespace(HfApi=lambda **kw:SimpleNamespace(model_info=lambda *a,**k:info,list_repo_files=lambda *a:['missing-test.safetensors']),hf_hub_download=forbidden)
    import socket
    socket.create_connection=forbidden
    spec=importlib.util.spec_from_file_location('owner_prompt_fetch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    terminal=sys.stdin.isatty()
    if not terminal:
        print('EVA_PROMPT_RESULT '+json.dumps(dict(status='SKIPPED',reason='No interactive terminal reached WSL',fixture=str(root))),flush=True);return 3
    offered=[];confirmed=[];answers=[];original=module.confirm_download
    # EOF also refuses a download, but is not an owner-entered decline.
    import builtins
    def observed_input(*args):
        answer=builtins.input(*args);answers.append(answer);return answer
    module.input=observed_input
    def observe():
        offered.append(True);answer=original();confirmed.append(answer);return answer
    module.confirm_download=observe
    os.environ.update(DEV_MODEL_OPERATION='prompt',DEV_UNATTENDED='')
    sys.argv=[str(path),'--allow-downloads',str(root/'models.txt'),str(weights),str(weights),str(root/'state')]
    print('OWNER CHECK: verify the file/size/source/licence and visible Type DOWNLOAD: prompt. Type NO, then Enter. No actual downloads are possible.',flush=True)
    code=module.main()
    owner_declined=len(answers)==1 and answers[0].strip()=='NO'
    passed=bool(offered) and confirmed==[False] and owner_declined and not calls and not list(weights.rglob('*')) and code==1
    result=dict(status='PASS' if passed else 'FAIL',offered=bool(offered),declined=confirmed==[False],
                owner_typed_no=owner_declined,input_eof=answers==[''],
                download_calls=len(calls),weight_files_created=len(list(weights.rglob('*'))),fetch_exit=code,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),fixture=str(root))
    print('EVA_PROMPT_RESULT '+json.dumps(result),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
