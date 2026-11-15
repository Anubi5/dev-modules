"""Check selected non-secret Hermes fields; separate policy errors from preferences.

This is configuration validation, not an OS security boundary. Shell tools share
the operator UID. Runtime acceptance also requires the existing network-off wall.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import os
from typing import Any
from urllib.parse import urlsplit
import hermes_runtime as runtime

# Built-in toolsets; an approved MCP server adds its id (Hermes aliases toolset mcp-<id> to <id>).
BASE_TOOLSETS=['terminal','file','skills','todo','clarify','memory','vision']
APPROVALS='eva-approved.json'   # in HERMES_HOME; written by integrations.py from the owner's decisions

EXPECTED={
    'security.allow_lazy_installs':False,
    'model.provider':'custom',
    'model.base_url':'http://127.0.0.1:8080/v1',
    'model.default':'qwen3.5-9b-agent',
    'model.context_length':65536,
    'platform_toolsets.cli':BASE_TOOLSETS,
    'plugins.enabled':[],
    'memory.provider':'none',
    'memory.memory_enabled':True,
    'memory.user_profile_enabled':False,
    'memory.write_approval':True,
    'skills.write_approval':True,
    'model.supports_vision':True,
    'agent.image_input_mode':'native',
    'auxiliary.vision.provider':'custom',
    'auxiliary.vision.base_url':'http://127.0.0.1:8080/v1',
    'auxiliary.vision.model':'qwen3.5-9b-agent',
    'auxiliary.compression.provider':'custom',
    'auxiliary.compression.base_url':'http://127.0.0.1:8080/v1',
    'auxiliary.compression.model':'qwen3.5-9b-agent',
    'auxiliary.approval.provider':'custom',
    'auxiliary.approval.base_url':'http://127.0.0.1:8080/v1',
    'auxiliary.approval.model':'qwen3.5-9b-agent',
    'auxiliary.background_review.enabled':False,
    'auxiliary.title_generation.enabled':False,
    'auxiliary.title_generation.model_upgrade_enabled':False,
}
DISABLED_SKILLS={'telegram','telegram-notify','telegram-client','hermes-telegram-business','send-message','browser-use','computer-use'}
ALLOWED_PLUGINS=set()   # module integrations arrive as declared MCP servers, never as Hermes plugins
PREFERENCES={'model.default','model.context_length','memory.memory_enabled',
             'model.supports_vision','agent.image_input_mode',
             'auxiliary.vision.model','auxiliary.compression.model','auxiliary.approval.model'}
# Never query entire fallback/provider objects: they can contain authentication.
FALLBACK_KEYS=tuple(prefix+'.'+leaf for prefix in ('fallback_model','fallback_providers.0')
                    for leaf in ('provider','base_url','model'))
READ_KEYS=tuple(EXPECTED)+('skills.disabled','mcp_servers')+FALLBACK_KEYS
UNSET_ALLOWED=FALLBACK_KEYS+('mcp_servers',)


def approved_servers(home: Path) -> dict[str,dict[str,Any]]:
    """MCP servers the owner approved: id -> {command, args}. A missing or unreadable file approves nothing."""
    try:
        data=json.loads((Path(home)/APPROVALS).read_text())
        decisions=data['decisions'] if data.get('schema')==1 else {}
    except (OSError,ValueError,KeyError,AttributeError):
        return {}
    servers={}
    for key,record in decisions.items():
        offer=record.get('offer') if isinstance(record,dict) else None
        if record.get('approved') is True and isinstance(offer,dict) and offer.get('kind')=='mcp' and key.startswith('mcp:'):
            servers[offer['id']]=dict(command=offer['command'],args=list(offer.get('args') or []))
    return servers


def local_endpoint(value: Any) -> bool:
    """Accept only literal loopback HTTP endpoints for the approved serving port."""
    try:
        url=urlsplit(value)
        return (url.scheme=='http' and url.hostname in ('127.0.0.1','::1') and url.port==8080
                and url.path.rstrip('/')=='/v1' and not url.username and not url.password
                and not url.query and not url.fragment)
    except (AttributeError,TypeError,ValueError):
        return False


def parse(text):
    text=text.strip()
    if not text:return None
    try:return json.loads(text)
    except ValueError:pass
    if text.startswith('- '):return [row.strip()[2:].strip().strip('\"\'') for row in text.splitlines()]
    return text


def evaluate(values: dict[str,Any],unreadable: tuple[str,...] | list[str]=(),approved: dict[str,dict[str,Any]] | None=None) -> dict[str,Any]:
    """Validate projected fields without returning config values or invoking tools.

    approved: MCP servers the owner approved (approved_servers); every configured server must match one exactly.
    """
    errors=[]; warnings=[]; approved=approved or {}
    toolsets=set(BASE_TOOLSETS)|set(approved)
    if values.get('security.allow_lazy_installs') is not False:
        errors.append('Runtime dependency installation is not approved')
    for key in unreadable:
        (warnings if key in PREFERENCES else errors).append('UNVERIFIED setting: '+key)
    for key in ('model.provider','auxiliary.vision.provider','auxiliary.compression.provider','auxiliary.approval.provider'):
        if values.get(key)!='custom':errors.append('Unapproved inference provider: '+key)
    for key in ('model.base_url','auxiliary.vision.base_url','auxiliary.compression.base_url','auxiliary.approval.base_url'):
        if not local_endpoint(values.get(key)):errors.append('Non-local or unapproved endpoint: '+key)
    for key,allowed,expected in [('platform_toolsets.cli',toolsets,toolsets),('plugins.enabled',ALLOWED_PLUGINS,set())]:
        value=values.get(key)
        if not isinstance(value,list) or not all(isinstance(x,str) for x in value) or not set(value).issubset(allowed):
            errors.append('Unapproved executable capabilities: '+key)
        elif set(value)!=expected:warnings.append('Expected capabilities missing: '+key)
    servers=values.get('mcp_servers')
    if servers not in (None,{}):
        if not isinstance(servers,dict):errors.append('Unapproved MCP servers: mcp_servers')
        else:
            for name,server in servers.items():
                want=approved.get(name)
                got=dict(command=server.get('command'),args=list(server.get('args') or [])) if isinstance(server,dict) else None
                if want is None or got!=want or set(server)-{'command','args','env','timeout','connect_timeout'}:
                    errors.append('Unapproved MCP server: '+str(name))
    if values.get('memory.provider') not in (None,'','none','builtin','built-in','default'):
        errors.append('External memory provider configured')
    if values.get('memory.user_profile_enabled') is not False:
        errors.append('Automatic user-profile collection is not approved')
    for key in ('memory.write_approval','skills.write_approval'):
        if values.get(key) is not True:errors.append('Cross-session writes require owner approval: '+key)
    for key in ('auxiliary.background_review.enabled','auxiliary.title_generation.enabled','auxiliary.title_generation.model_upgrade_enabled'):
        if values.get(key) is not False:errors.append('Unapproved background inference: '+key)
    disabled=values.get('skills.disabled')
    if not isinstance(disabled,list) or not all(isinstance(x,str) for x in disabled) or not DISABLED_SKILLS.issubset(disabled):errors.append('Disabled skill policy changed')
    for key in FALLBACK_KEYS:
        if values.get(key) not in (None,''):errors.append('Fallback inference configured: '+key)
    for key in PREFERENCES:
        if values.get(key)!=EXPECTED[key]:warnings.append('Preference drift: '+key)
    return dict(schema='local-ai.hermes-profile/1',errors=sorted(set(errors)),warnings=sorted(set(warnings)))


def inspect(executable: str,soul: Path,env: dict[str,str] | None=None) -> dict[str,Any]:
    """Check named non-secret leaves using the official JSON output contract.

    Args:
        executable: Existing Hermes launcher.
        soul: Selected profile's SOUL file, used only for the owned-section check.
        env: Optional environment to which the offline runtime policy is applied.

    Returns:
        Policy errors and preference warnings. Invalid JSON is unverified;
        display formatting never changes the actual value being checked.
    """
    env=runtime.environment(env)
    values={}; unreadable=[]
    deadline=time.monotonic()+60
    for key in READ_KEYS:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            unreadable.extend(READ_KEYS[len(values)+len(unreadable):]);break
        try:
            result=subprocess.run([executable,'config','get','--json',key],capture_output=True,text=True,timeout=min(20,remaining),env=env)
            missing='Config key not set: '+key
            if result.returncode and key in UNSET_ALLOWED and missing in (result.stdout+result.stderr):
                values[key]=None # Official CLI explicitly reports an unset leaf, not a transport failure.
            elif result.returncode or len(result.stdout)>65536:unreadable.append(key)
            else:values[key]=json.loads(result.stdout)
        except (OSError,ValueError,subprocess.TimeoutExpired):unreadable.append(key)
    report=evaluate(values,unreadable,approved_servers(soul.parent))
    template=Path(__file__).with_name('hermes-local-SOUL.md').read_text()
    text=soul.read_text() if soul.is_file() else ''
    if template.strip() not in text:report['warnings'].append('Owned SOUL profile section changed or missing')
    return report


def check(executable: str,soul: Path) -> list[str]:
    """Return policy failures for compatibility with the existing checker interface."""
    return inspect(executable,soul)['errors']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes',default='hermes')
    parser.add_argument('--soul',type=Path,default=Path(os.environ.get('HERMES_HOME',Path.home()/'.hermes'))/'SOUL.md')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    try:
        runtime.verify_installed(args.hermes)
        report=inspect(args.hermes,args.soul)
    except (OSError,ValueError,subprocess.SubprocessError) as error:report=dict(schema='local-ai.hermes-profile/1',errors=[type(error).__name__+' during profile check; run the reviewed Hermes install'],warnings=[])
    if args.json:print(json.dumps(report))
    else:
        for error in report['errors']:print('CHECK Hermes profile: '+error)
        for warning in report['warnings']:print('WARN Hermes profile: '+warning)
        if not report['errors']:print('OK Hermes selected local-policy fields (preferences reported separately; not an isolation boundary)')
    return 1 if report['errors'] else 0


if __name__=='__main__':raise SystemExit(main())
