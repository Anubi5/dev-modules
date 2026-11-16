#!/usr/bin/env python3
"""Run reviewed CPU-only test entries. Acceptance is always an explicit separate command."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def worker(path, result_path, collect):
    """Isolate module imports and report actual unittest discovery, skips and failures."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    def identities(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from identities(item)
            else:
                yield '.'.join(item.id().split('.')[-2:])
    names = sorted(identities(suite))
    result = dict(discovered=len(names), executed=0, identities=names, skipped=[], failures=0, errors=0, expected_failures=0, unexpected_successes=0)
    if not collect:
        outcome = unittest.TextTestRunner(verbosity=1).run(suite)
        result.update(skipped=[dict(test=t.id(), reason=why) for t, why in outcome.skipped],
                      failures=len(outcome.failures), errors=len(outcome.errors), executed=outcome.testsRun,
                      expected_failures=len(outcome.expectedFailures), unexpected_successes=len(outcome.unexpectedSuccesses))
    result_path.write_text(json.dumps(result), encoding='utf-8')
    return bool(result['failures'] or result['errors'] or result['unexpected_successes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', action='append', help='owner name; repeat or comma-separate (default all)')
    parser.add_argument('--suite', choices=('fast', 'integration'), default='fast')
    parser.add_argument('--strict', action='store_true', help='fail release verification on any skipped, partial or empty execution')
    parser.add_argument('--list', action='store_true', help='show entries and prerequisites without importing tests')
    parser.add_argument('--collect', action='store_true', help='collect unittest identities without running cases')
    parser.add_argument('--report', type=Path, help='write machine-readable results')
    parser.add_argument('--match', default='', help='run only entries containing this path/name')
    parser.add_argument('--powershell', help='existing native PowerShell executable')
    parser.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        return worker(args._worker, args._result, args.collect)
    rows = json.loads((ROOT/'tests/inventory.json').read_text(encoding='utf-8'))['tests']
    registered={r['path'] for r in rows}
    for folder in [ROOT/'tests', *[ROOT/name/'tests' for name in ('agent-hermes','blender','character','comfyui','llama-swap','reverse-tools','rig')]]:
        for category in ('unit','contract','integration','acceptance'):
            for path in (folder/category).rglob('*'):
                if path.suffix in ('.py','.sh','.ps1') and (path.name.startswith('test_') or '_test.' in path.name or path.suffix in ('.sh','.ps1')) and path.relative_to(ROOT).as_posix() not in registered:
                    parser.error('unregistered test: '+str(path)+'; classify it in tests/inventory.json')
    absent = [r['path'] for r in rows if not (ROOT/r['path']).is_file()]
    if absent:
        parser.error('inventoried test file missing: '+', '.join(absent))
    owners = {v for item in args.module or ['all'] for v in item.split(',')}
    known = {'all','core','agent-hermes','blender','character','comfyui','llama-swap','reverse-tools','rig'}
    if owners-known:
        parser.error('unknown module: '+', '.join(sorted(owners-known)))
    categories = {'unit','contract'} if args.suite == 'fast' else {'integration'}
    selected = [r for r in rows if r['entrypoint'] and r['category'] in categories
                and args.match in r['path']
                and ('all' in owners or r['owner'] in owners or owners.intersection(r.get('modules',[])))]
    if not selected:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(dict(selected=0, statuses={}, reason='NO TESTS selected'))+'\n', encoding='utf-8')
        print('NO TESTS selected; use --list or choose a different module/suite')
        return 2
    if args.list:
        for row in selected:
            print(f"{row['owner']}/{row['category']}: {row['path']} [{row['platform']}; {', '.join(row['requires']) or 'standard tools'}]")
        return 0
    started = time.monotonic()
    results = []
    for row in selected:
        item = dict(id=row['id'], path=row['path'], owner=row['owner'], category=row['category'])
        missing = []
        if row['platform']=='linux' and not sys.platform.startswith('linux'):
            missing.append('requires an existing Linux environment; runner never starts WSL')
        if row['platform']=='windows' and sys.platform!='win32':
            missing.append('requires native Windows')
        platform_skip = bool(missing)
        missing += ['missing Python package '+p for p in row['requires'] if importlib.util.find_spec(p) is None]
        path = ROOT/row['path']
        if not path.is_file():
            item.update(status='FAIL',reason='inventoried test file missing'); results.append(item); print(item,flush=True);continue
        if missing:
            item.update(status='SKIP',skip_kind='platform' if platform_skip else 'dependency',reason='; '.join(missing));results.append(item)
            print('SKIP',row['path'],item['reason'],flush=True);continue
        with tempfile.TemporaryDirectory(prefix='eva-tests-') as temporary:
            result_path=Path(temporary)/'result.json'
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', UV_OFFLINE='1', PIP_NO_INDEX='1',
                     HERMES_DISABLE_LAZY_INSTALLS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            home=Path(temporary)/'home'
            home.mkdir()
            env.update(HOME=str(home),USERPROFILE=str(home),HERMES_HOME=str(home/'.hermes'),
                       XDG_CACHE_HOME=str(home/'.cache'),XDG_CONFIG_HOME=str(home/'.config'),XDG_STATE_HOME=str(home/'.local/state'),
                       MODELS_MOUNT=str(home/'models'),MODELS_RW=str(home/'models-rw'))
            owner=ROOT/row['owner'] if row['owner']!='core' else ROOT
            paths=[path.parent, owner, ROOT, ROOT/'tests/fixtures']
            paths += [owner/'tests'/category for category in ('unit','contract','integration','fixtures','acceptance')]
            env['PYTHONPATH']=os.pathsep.join([*map(str,paths),os.environ.get('PYTHONPATH','')])
            if path.suffix=='.py' and row['case_names']:
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'--_worker',str(path),'--_result',str(result_path)]
                if args.collect: command.append('--collect')
            elif args.collect:
                item.update(status='COLLECTED',identities=['<script entry>'],discovered=1);results.append(item);continue
            elif path.suffix=='.py':
                command=[sys.executable,'-B',str(path)]
            elif path.suffix=='.sh':
                command=['/bin/bash',str(path)]  # .sh entries require Linux; never a Windows WSL alias.
            else:
                shell=args.powershell or shutil.which('pwsh') or shutil.which('powershell')
                if not shell or 'windowsapps' in shell.lower():
                    item.update(status='SKIP',skip_kind='dependency',reason='native PowerShell parser/runtime unavailable');results.append(item);continue
                command=[shell,'-NoProfile','-NonInteractive','-File',str(path)]
                if path.name in ('cli_transport_test.ps1','progress_test.ps1'):
                    command+=['-NativePython',sys.executable]
            start=time.monotonic()
            try:
                child=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
                output=child.stdout+child.stderr
                item.update(status='PASS' if child.returncode==0 else 'FAIL',exit_code=child.returncode)
                if result_path.exists():
                    item.update(json.loads(result_path.read_text()))
                    item['passed_cases'] = max(0, item['executed'] - len(item['skipped']) - item['failures'] - item['errors'] - item['expected_failures'] - item['unexpected_successes']) if not args.collect else 0
                    if child.returncode==0:
                        item['status']='COLLECTED' if args.collect else 'PARTIAL' if item['skipped'] or item['executed'] != item['discovered'] else 'PASS'
                    if not item['discovered']:
                        item.update(status='FAIL',reason='zero unittest cases discovered')
                    if set(item['identities']) != set(row['case_names']):
                        item.update(status='FAIL',reason='test identities differ from the classified inventory')
                elif row['case_names']:
                    item.update(status='FAIL',reason='unittest worker result missing')
                elif child.returncode==0 and re.search(r'^\s*(SKIP|STUB)\b|OK \(skipped=[1-9]',output,re.M|re.I):
                    item.update(status='SKIP',skip_kind='child',reason='child explicitly reported SKIP/STUB; not accepted as PASS')
                item['output']=output
            except subprocess.TimeoutExpired as error:
                item.update(status='FAIL',reason='CPU fixture exceeded 180 seconds',output=str(error))
            item['elapsed_seconds']=round(time.monotonic()-start,3)
            results.append(item)
            print(item['status'],row['path'],f"{item['elapsed_seconds']:.3f}s",flush=True)
            if item['status'] in ('FAIL','PARTIAL'):
                print(item.get('output',''),flush=True)
    summary=dict(suite=args.suite,strict=args.strict,platform=sys.platform,elapsed_seconds=round(time.monotonic()-started,3),selected=len(selected),
                 statuses={s:sum(r['status']==s for r in results) for s in ('PASS','FAIL','PARTIAL','SKIP','COLLECTED')},
                 discovered_cases=sum(r.get('discovered',0) for r in results),results=results)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
    return int(any(r['status']=='FAIL' or (args.strict and r['status'] in ('SKIP','PARTIAL','COLLECTED')) for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
