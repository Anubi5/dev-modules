"""Prepared supervised Hermes acceptance; requires separate live-test approval.

Uses a private test HERMES_HOME with synthetic memory, never the operator's
personal memory or credentials. Requires the existing network-off wall and an
already-active llama-swap. Does not install dependencies or start services.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hermes_local_setup as setup
import hermes_profile_check as policy
import hermes_runtime as runtime
from agent_measurements import gpu_idle_gate


def events_result(text,tool=None):
    """Require the official stream protocol and successful tool completion."""
    events=[json.loads(line) for line in text.splitlines() if line.strip()]
    initial=[event for event in events if event.get('type')=='system' and event.get('subtype')=='init']
    results=[event for event in events if event.get('type')=='result']
    if len(initial)!=1 or initial[0].get('model')!='qwen3.5-9b-agent':raise ValueError('Hermes selected an unexpected model')
    if len(results)!=1 or results[0].get('exit_code')!=0:raise ValueError('Hermes did not produce one successful result')
    if tool:
        if not any(e.get('type')=='tool_use' and e.get('name')==tool for e in events):raise ValueError('Required tool was not invoked: '+tool)
        if not any(e.get('type')=='tool_result' and e.get('name')==tool and e.get('is_error') is False for e in events):
            raise ValueError('Required tool did not complete successfully: '+tool)
    return results[0]['text']


def nested_settings():
    """Project reviewed non-secret settings into an isolated YAML-compatible JSON file."""
    output={}
    for key,value in setup.settings().items():
        target=output;parts=key.split('.')
        for part in parts[:-1]:target=target.setdefault(part,{})
        target[parts[-1]]=value
    output['plugins']['enabled']=[] # This fixture tests builtin memory/vision, not plugin installation.
    output['model']['api_key']='local' # Synthetic local-client placeholder, never an existing credential.
    output['fallback_model']=None;output['fallback_providers']=[]
    return output


def service_states():
    """Inspect only the services affected by this bounded acceptance."""
    result={}
    for unit,user in [('llama-swap',True),('comfyui',True),('hermes-dashboard',True),('net-off',False)]:
        command=['systemctl']+(['--user'] if user else [])+['is-active',unit]
        p=subprocess.run(command,capture_output=True,text=True,timeout=10)
        result[unit]=p.stdout.strip()
    return result


def hermes_env(home):
    """Select public runtime variables, never third-party credential variables."""
    # Do not copy third-party credential variables or personal Hermes state.
    env={key:os.environ[key] for key in ('PATH','HOME','USER','LANG','TERM','XDG_RUNTIME_DIR') if key in os.environ}
    env.update(HERMES_HOME=str(home),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',UV_OFFLINE='1',PIP_NO_INDEX='1',OPENAI_API_KEY='local')
    return runtime.environment(env)


def run_agent(executable,home,work,prompt,label,image=None,tool=None,toolsets=None):
    """Run a fresh Hermes process with a small environment; retain private JSONL evidence."""
    env=hermes_env(home)
    # Same production entry point; no test-only dependency/runtime selection.
    command=['ai','hermes','chat','-Q','-q',prompt,'--format','stream-json','--max-turns','8','--toolsets',toolsets or ('vision' if image else 'memory,terminal')]
    if image:command+=['--image',str(image)]
    log=work/(label+'.jsonl');errorlog=work/(label+'.stderr')
    with log.open('x') as stdout,errorlog.open('x') as stderr:
        process=subprocess.Popen(command,cwd=work,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            code=process.wait(timeout=900)
            if code:raise RuntimeError('Hermes invocation failed: '+label)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    return events_result(log.read_text(),tool)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--hermes',default='hermes')
    parser.add_argument('--image',type=Path,required=True,help='owner-approved real local test image, outside model weights')
    parser.add_argument('--expected-vision',required=True,help='independently chosen phrase for a smoke check; owner must also review the answer')
    parser.add_argument('--hermes-source',type=Path,required=True)
    parser.add_argument('--idle-baseline',type=int,default=0,help='owner-approved background GPU percent; default remains zero, allowance remains 5%')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true',help='create a private fixture and stop; dependencies remain an owner install operation')
    mode.add_argument('--resume',action='store_true',help='use the previously prepared private fixture after owner dependency preparation')
    args=parser.parse_args(argv)
    if os.name!='posix':parser.error('Run only inside an already-running approved Linux environment')
    image=args.image.resolve(strict=True)
    args.out=args.out.resolve()
    if args.out.is_relative_to('/models'):parser.error('Acceptance output must be outside protected models')
    if image.suffix.lower() not in ('.png','.jpg','.jpeg','.webp') or image.is_relative_to('/models'):
        parser.error('Use a reviewed local image, never a model file')
    before=service_states()
    if before['net-off']!='active' or before['llama-swap']!='active' or before['comfyui'] not in ('inactive','failed'):
        raise RuntimeError('Requires network OFF, llama-swap already active and ComfyUI already stopped; nothing changed')
    revision=runtime.verify_installed(args.hermes,args.hermes_source)['revision']
    idle=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    gpu_gate=gpu_idle_gate(int(idle.stdout.strip()),args.idle_baseline)
    if args.resume:
        if not args.out.is_dir() or args.out.is_symlink():raise ValueError('Resume requires an existing private prepared fixture')
    else:args.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    previous_umask=os.umask(0o077)
    home=args.out/'hermes-home'
    if not args.resume:
        home.mkdir(mode=0o700)
        (home/'config.yaml').write_text(json.dumps(nested_settings(),indent=2))
        (home/'SOUL.md').write_text(Path(__file__).resolve().parents[1].joinpath('hermes-local-SOUL.md').read_text())
    elif home.is_symlink() or not (home/'config.yaml').is_file():raise ValueError('Prepared private config missing')
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED',home=str(home),gpu_gate=gpu_gate,next='Owner must prepare pinned dependencies for this HERMES_HOME during an approved install, then resume; no live acceptance claimed')))
        os.umask(previous_umask);return 0
    result=dict(schema='local-ai.hermes-acceptance/1',status='running',hermes_revision=revision,services_before=before,gpu_gate=gpu_gate,
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),cases={},vision_owner_review='required')
    try:
        result['profile']=policy.inspect(args.hermes,home/'SOUL.md',env=hermes_env(home))
        if result['profile']['errors']:raise ValueError('Isolated Hermes profile failed compatibility/security preflight')
        setup.private_memory(home)
        import fcntl
        with open('/tmp/ai-gpu.lock','a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            key='eva_probe_'+uuid.uuid4().hex[:10];token=uuid.uuid4().hex
            answer=run_agent(args.hermes,home,args.out,'What is 17 times 23? Return only the number.','question')
            if answer.strip()!='391':raise ValueError('Question answer did not match independent arithmetic')
            result['cases']['question']='PASS'
            marker=args.out/'tool-marker.txt'
            run_agent(args.hermes,home,args.out,'Use terminal to run exactly this command:\nprintf %s '+shlex.quote(token)+' > '+shlex.quote(str(marker))+'\nThen reply DONE.','tool',tool='terminal')
            if not marker.is_file() or marker.read_text()!=token:raise ValueError('Terminal output marker missing or incorrect')
            result['cases']['local_tool']='PASS'
            run_agent(args.hermes,home,args.out,'Call the memory tool EXACTLY ONCE to add this synthetic test fact to the memory target: '+key+' = '+token+'. A staged-for-approval result is success for this request: reply PENDING and stop. Do not retry, patch, approve, or store a user profile.','memory-write',tool='memory',toolsets='memory')
            memory=home/'memories/MEMORY.md'
            if memory.exists() and token in memory.read_text():raise ValueError('Memory was committed without the required owner approval')
            print('Synthetic memory write staged. In a second terminal run: HERMES_HOME='+shlex.quote(str(home))+' ai hermes')
            print('Use /memory pending, review and approve ONLY this synthetic fact: '+key+' = '+token+'.')
            if input('Type CONTINUE after approval, or anything else to cancel: ').strip()!='CONTINUE':raise KeyboardInterrupt('Owner did not approve synthetic memory write')
            if not memory.is_file() or token not in memory.read_text():raise ValueError('Built-in memory file did not persist the synthetic fact')
            memory_before=hashlib.sha256(memory.read_bytes()).hexdigest()
            recalled=run_agent(args.hermes,home,args.out,'Read the value for '+key+' from memory already provided in your context. Do not call tools, write or patch memory. Return the value only, without Markdown or explanation.','memory-recall',toolsets='memory')
            if recalled.strip()!=token:raise ValueError('Fresh-process memory recall failed')
            if hashlib.sha256(memory.read_bytes()).hexdigest()!=memory_before:raise ValueError('Read-only recall changed memory')
            if memory.stat().st_mode&0o077 or memory.parent.stat().st_mode&0o077:raise ValueError('Memory permissions are not private')
            user=home/'memories/USER.md'
            if user.exists() and user.stat().st_size:raise ValueError('Unexpected automatic user-profile collection')
            result['cases']['memory_persistence_restart']='PASS'
            description=run_agent(args.hermes,home,args.out,'Describe the attached image briefly, identifying its main subject and colours.','vision',image=image)
            result['vision_answer']=description
            if args.expected_vision.casefold() not in description.casefold():raise ValueError('Image description failed the independent smoke expectation')
            if hashlib.sha256(image.read_bytes()).hexdigest()!=result['image_sha256']:raise ValueError('Input image changed')
            result['cases']['hermes_vision_smoke']='PASS'
            result['status']='smoke_passed_owner_review_required'
    except KeyboardInterrupt:result['status']='CANCELLED'
    except Exception as error:result.update(status='FAIL',error=type(error).__name__+': '+str(error))
    finally:
        result['services_after']=service_states()
        result['services_unchanged']=before==result['services_after']
        if not result['services_unchanged']:result['status']='FAIL'
        (args.out/'result.json').write_text(json.dumps(result,indent=2))
        os.umask(previous_umask)
    print(json.dumps(result))
    return 0 if result['status']=='smoke_passed_owner_review_required' else 130 if result['status']=='CANCELLED' else 1


if __name__=='__main__':
    def stop(signum,frame):raise KeyboardInterrupt('Acceptance cancelled')
    signal.signal(signal.SIGTERM,stop)
    raise SystemExit(main())
