"""Supervised 64K agent benchmark with/without vision; no model/config downloads.

This starts isolated llama-server processes and restores the prior chat service.
It is not Hermes vision acceptance. Shared Windows GPU memory needs a separate
host counter and is explicitly unmeasured here. Do not execute as an offline test.
"""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

from agent_measurements import NoRedirect, red_png, stream_chat, gpu_idle_gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include-daily',action='store_true',help='optional 32K comparison; daily serving configuration is never changed')
    parser.add_argument('--idle-baseline',type=int,default=0,help='owner-approved background GPU percent; default remains zero, allowance remains 5%')
    parser.add_argument('--agent-context',type=int,choices=[8192,16384,32768,65536],default=65536,
                        help='measure a practical agent context when 64K is unsuitable; never changes installed serving configuration')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    module=Path.home()/'.local/share/modules/llama-swap'
    record=json.loads((module/'models/models.json').read_text())['qwen3.5-9b']
    config=json.loads((module/'models.json').read_text())
    base=next(p for p in config['chat'] if p['id']=='qwen3.5-9b')
    paths=record['files']+([record['mmproj']] if record.get('mmproj') else [])
    assert all(Path(p).resolve().is_relative_to('/models/llm') for p in paths)
    before={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}
    assert os.statvfs('/models').f_flag & os.ST_RDONLY
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    url='http://127.0.0.1:'+str(port)
    result=dict(weight_files=paths,same_file=True,profiles={},shared_gpu_memory_mib=None,
                limitations=['Windows shared GPU memory is not measured by nvidia-smi; collect host counters separately',
                             'Fresh server load is not a flushed OS disk-cache measurement'])
    comfy=subprocess.run(['systemctl','--user','is-active','comfyui'],capture_output=True,text=True).stdout.strip()
    if comfy not in ['inactive','failed']:raise RuntimeError('ComfyUI must already be stopped; benchmark will not stop it')
    idle=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    result['gpu_gate']=gpu_idle_gate(int(idle.stdout.strip()),args.idle_baseline)
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    state=subprocess.run(['systemctl','--user','is-active','llama-swap'],capture_output=True,text=True).stdout.strip()
    if state not in ['active','inactive','failed']:raise RuntimeError('Unknown prior chat state')
    with open('/tmp/ai-gpu.lock','a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if state=='active':subprocess.run(['systemctl','--user','stop','llama-swap'],check=True)
            profiles=[('qwen3.5-9b-agent',args.agent_context)]
            if args.include_daily:profiles.insert(0,('qwen3.5-9b',32768))
            cases=[(name,context,vision) for name,context in profiles for vision in (False,True)]
            for name,context,vision in cases:
                case=name+('-vision' if vision else '-text')
                if vision and not record.get('mmproj'):raise RuntimeError('Matching daily-model projector is missing; no substitute/download allowed')
                command=[str(module/'app/bin/llama-server'),'--host','127.0.0.1','--port',str(port),'--jinja',
                         '-ngl','99','-m',record['file'],'-c',str(context),'-ctk','q8_0','-ctv','q8_0']
                if vision:command+=['--mmproj',record['mmproj']]
                samples=[];stop=threading.Event()
                def monitor():
                    while not stop.is_set():
                        p=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                        if p.returncode==0:
                            try:
                                sample=dict(zip(['used_mib','total_mib','utilization'],map(int,p.stdout.strip().split(','))))
                                if profile.get('pid'):
                                    status=Path('/proc')/str(profile['pid'])/'status'
                                    if status.exists():
                                        match=re.search(r'^VmRSS:\s+(\d+)',status.read_text(),re.M)
                                        sample['process_rss_kib']=int(match[1]) if match else None
                                samples.append(sample)
                            except (OSError,ValueError):pass
                        stop.wait(.5)
                profile=dict(context=context,command=command,vision=vision,cache_types=dict(key='q8_0',value='q8_0'),requests=[])
                thread=threading.Thread(target=monitor);thread.start()
                result['profiles'][case]=profile
                started=time.monotonic()
                with (args.out/(case+'.log')).open('w') as log:
                    process=None
                    try:
                        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        profile['pid']=process.pid
                        while True:
                            if process.poll() is not None:raise RuntimeError('Isolated model server exited before health')
                            try:
                                with http.open(url+'/health',timeout=2):break
                            except OSError:
                                if time.monotonic()-started>240:raise TimeoutError('Model load exceeded 240 seconds')
                                time.sleep(1)
                        profile['load_seconds']=time.monotonic()-started
                        with http.open(url+'/props',timeout=5) as response:props=json.load(response)
                        profile['reported_context']=props.get('default_generation_settings',{}).get('n_ctx')
                        profile['reported_model_path']=props.get('model_path')
                        for attempt in range(2):
                            content=([dict(type='text',text='What is the dominant colour? Describe briefly.'),
                                      dict(type='image_url',image_url=dict(url='data:image/png;base64,'+red_png()))]
                                     if vision else 'List twenty simple colours, separated by commas.')
                            measurement=stream_chat(url,name,content,opener=http)
                            assert measurement['text'] and measurement['first_token_seconds'] is not None
                            if vision:assert 'red' in measurement['text'].lower(),'Direct server failed the synthetic image check'
                            profile['requests'].append(measurement)
                        profile['status']='passed'
                    except KeyboardInterrupt:
                        profile.update(status='interrupted');raise
                    except Exception as error:
                        profile.update(status='failed',error=str(error))
                    finally:
                        if process is not None:
                            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                        stop.set();thread.join(timeout=10)
                        profile['gpu_samples']=samples
                        profile['peak_vram_mib']=max((p['used_mib'] for p in samples),default=None)
                        profile['peak_process_rss_kib']=max((p['process_rss_kib'] for p in samples if p.get('process_rss_kib') is not None),default=None)
                        text=(args.out/(case+'.log')).read_text(errors='replace')
                        file_type=re.findall(r'general\.file_type[^\n]*?=\s*(\d+)',text)
                        offload=re.findall(r'offloaded (\d+)/(\d+) layers',text)
                        profile['gguf_file_type']=int(file_type[-1]) if file_type else None
                        profile['offload_layers']=list(map(int,offload[-1])) if offload else None
                        (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(case,profile['status'],'peak VRAM MiB',profile['peak_vram_mib'],flush=True)
        finally:
            try:
                if state=='active':subprocess.run(['systemctl','--user','start','llama-swap'],check=True)
            finally:
                restored=subprocess.run(['systemctl','--user','is-active','llama-swap'],capture_output=True,text=True).stdout.strip()
                result['service_restoration']=dict(before=state,after=restored,matched=state==restored)
                (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    after={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}
    assert before==after
    result['weights_unchanged']=True
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['service_restoration']['matched'] or any(p['status']!='passed' for p in result['profiles'].values()):raise SystemExit(1)


if __name__=='__main__':
    def stop(signum,frame):raise KeyboardInterrupt('Benchmark cancelled')
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
