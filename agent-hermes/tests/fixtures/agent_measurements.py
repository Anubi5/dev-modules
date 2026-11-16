"""Local-only streaming measurements; missing counters stay unmeasured."""
import base64
import json
import struct
import time
import urllib.request
import zlib
from urllib.parse import urlsplit


def gpu_idle_gate(observed, baseline=0):
    """Keep the default <=5% gate; record a separately owner-approved baseline."""
    if type(baseline) is not int or not 0 <= baseline <= 95:
        raise ValueError('GPU baseline must be an explicitly approved integer from 0 to 95')
    if type(observed) is not int or not 0 <= observed <= 100:
        raise ValueError('Invalid GPU utilization sample')
    gate=dict(observed_percent=observed,baseline_percent=baseline,allowance_percent=5,limit_percent=baseline+5)
    if observed > gate['limit_percent']:
        raise RuntimeError('GPU exceeds the approved baseline plus 5%: '+str(gate))
    return gate


def red_png():
    """Return a synthetic image fixture without model downloads or image libraries."""
    raw=b''.join(b'\0'+b'\xff\0\0'*64 for _ in range(64))
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    image=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    return base64.b64encode(image).decode('ascii')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('Benchmark endpoint redirects are not approved')


def stream_chat(url,model,content,timeout=180,clock=time.monotonic,opener=None):
    """Measure first generated delta and server-reported token counts via SSE.

    Role/usage-only events do not count as a first token. A missing usage counter
    is None, never an invented token count or zero-throughput result.
    """
    parsed=urlsplit(url)
    if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Benchmark requires a literal loopback HTTP endpoint')
    opener=opener or urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    payload=dict(model=model,messages=[dict(role='user',content=content)],stream=True,
                 stream_options=dict(include_usage=True),temperature=0,max_tokens=96,
                 chat_template_kwargs=dict(enable_thinking=False))
    request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=clock();first=None;last=None;text=[];usage={};timings=None;served=set()
    with opener.open(request,timeout=timeout) as response:
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':break
            event=json.loads(data)
            if event.get('model'):served.add(event['model'])
            if event.get('usage'):usage=event['usage']
            if event.get('timings'):timings=event['timings']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('content') or delta.get('reasoning_content'):
                    tick=clock();first=tick if first is None else first;last=tick
                if delta.get('content'):text.append(delta['content'])
    elapsed=clock()-start
    if len(served)>1:raise ValueError('Server changed model identity within one response')
    tokens=usage.get('completion_tokens');duration=(last-first) if first is not None else None
    return dict(text=''.join(text),served_model=next(iter(served),None),seconds=elapsed,
                first_token_seconds=(first-start) if first is not None else None,
                prompt_tokens=usage.get('prompt_tokens'),completion_tokens=tokens,completion_tokens_per_generation_second=(tokens-1)/duration if isinstance(tokens,int) and tokens>1 and duration else None,
                server_timings=timings)
