"""CPU-only 3D subprocess routing, deadline and recovery regressions."""
import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Model3DDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.rt = SimpleNamespace(JOB={}, GpuBusy=type('GpuBusy',(Exception,),{}),
                                  generate=MagicMock(), job_state=MagicMock(),
                                  chat_pause=MagicMock(return_value=True), chat_resume=MagicMock())
        self.released = False
        @contextlib.contextmanager
        def lease(**kwargs):
            try: yield SimpleNamespace(fileno=lambda: 17)
            finally: self.released = True
        self.rt.gpu_lease = MagicMock(side_effect=lease)
        with patch.dict(sys.modules, runtime=self.rt), patch.object(sys, 'path', [str(ROOT),*sys.path]):
            self.model = load('deadline_model3d', ROOT/'operations/model3d.py')
            self.acceptance = load('acceptance_routing', ROOT/'tests/acceptance/comfy_test.py')
        self.now = 100.0
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.model.time,'monotonic',side_effect=lambda:self.now))
        self.stack.enter_context(patch.object(self.model,'glb_stats',return_value={'valid':True,'triangles':10}))
        self.stack.enter_context(patch.object(self.model.shutil,'which',return_value='fixture-blender'))
        self.stack.enter_context(patch.object(self.model,'handover_ok',return_value=True))
        self.stack.enter_context(patch.object(self.model.os.path,'exists',return_value=True))
        self.info = {}

    def generation(self, *args, **kwargs):
        self.now += 4
        return True, ['fixture.glb'], ''

    def blender(self):
        return SimpleNamespace(returncode=0,stdout='MESH-CHECK '+json.dumps({'ok':True,'renders':['a','b','c','d']}),stderr='')

    def test_acceptance_uses_public_generation_cli_and_preserves_result(self):
        result=SimpleNamespace(returncode=0,stdout='fixture.glb\n  model: 10 triangles',stderr='')
        with patch.object(subprocess,'run',return_value=result) as child:
            self.assertIs(self.acceptance.generate_3d('input.png'),result)
        argv=child.call_args.args[0]
        self.assertEqual(Path(argv[1]), ROOT/'gen.py')
        self.assertEqual(argv[2:5], ['3d','--image','input.png'])
        self.assertIn('--timeout', argv)
        self.assertEqual(child.call_args.kwargs['timeout'],2700)

    def test_success_shares_remaining_budget_across_generation_and_checks(self):
        self.rt.generate.side_effect=self.generation
        calls=[]
        def child(argv, **kwargs):
            calls.append(kwargs['timeout'])
            self.now+=2
            return self.blender() if len(calls)==1 else SimpleNamespace(returncode=0,stdout='OK')
        with patch.object(subprocess,'run',side_effect=child):
            status,files,error=self.model.job_3d({},[],self.info,10,log=lambda _:None)
        self.assertEqual(status,'succeeded')
        self.assertEqual(calls,[6,4])
        self.assertEqual(self.rt.generate.call_args.kwargs['deadline'],110)
        self.assertEqual(self.rt.gpu_lease.call_args.kwargs['wait'],10)
        self.assertTrue(self.released)
        self.rt.chat_resume.assert_called_once_with(True, unittest.mock.ANY)

    def test_blender_timeout_recovers_chat_and_releases_lock(self):
        self.rt.generate.side_effect=self.generation
        with patch.object(subprocess,'run',side_effect=subprocess.TimeoutExpired('blender',6)):
            status,files,error=self.model.job_3d({},[],self.info,10,log=lambda _:None)
        self.assertEqual(status,'check_failed')
        self.assertTrue(self.released)
        self.assertIn('timed out',self.info['blender']['error'])
        self.rt.chat_resume.assert_called_once()
        self.assertNotEqual(self.model.EXIT[status],0)

    def test_image_checker_timeout_records_failure_and_releases_lock(self):
        self.rt.generate.side_effect=self.generation
        with patch.object(subprocess,'run',side_effect=[self.blender(),subprocess.TimeoutExpired('img-check',6)]):
            status,files,error=self.model.job_3d({},[],self.info,10,log=lambda _:None)
        self.assertEqual(status,'check_failed')
        self.assertIn('TimeoutExpired',error)
        self.assertTrue(self.released)
        self.rt.chat_resume.assert_called_once()
        self.assertEqual(self.rt.job_state.call_args.kwargs['error'],error)

    def test_expired_generation_never_launches_validation(self):
        def late(*args,**kwargs):
            self.now+=11
            return True,['fixture.glb'],''
        self.rt.generate.side_effect=late
        with patch.object(subprocess,'run') as child:
            status,_,error=self.model.job_3d({},[],self.info,10,log=lambda _:None)
        child.assert_not_called()
        self.assertEqual(status,'check_failed')
        self.assertIn('deadline exceeded',error)
        self.assertTrue(self.released)

    def test_pause_timeout_attempts_recovery(self):
        self.rt.generate.side_effect=self.generation
        self.rt.JOB['chat_paused']=True
        self.rt.chat_pause.side_effect=subprocess.TimeoutExpired('systemctl',6)
        status,_,_=self.model.job_3d({},[],self.info,10,log=lambda _:None)
        self.assertEqual(status,'check_failed')
        self.rt.chat_resume.assert_called_once_with(True,unittest.mock.ANY)
        self.assertTrue(self.released)

class ServiceBudgetTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((ROOT/'runtime.py').read_text())
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('budget','chat_pause','chat_resume','_run')]
        self.ns=dict(time=time,subprocess=subprocess,JOB={})
        exec(compile(tree,'runtime.py','exec'),self.ns)

    def test_service_calls_have_remaining_timeout_and_failed_resume_retains_marker(self):
        self.ns['JOB']['chat_paused']=True
        with patch.object(subprocess,'run',side_effect=subprocess.TimeoutExpired('systemctl',30)) as child:
            with self.assertRaises(subprocess.TimeoutExpired): self.ns['chat_resume'](True,lambda _:None)
            self.assertEqual(child.call_args.kwargs['timeout'],30)
        self.assertTrue(self.ns['JOB']['chat_paused'])
        with patch.object(subprocess,'run',return_value=SimpleNamespace(returncode=0)) as child:
            self.ns['chat_resume'](True,lambda _:None)
        self.assertNotIn('chat_paused',self.ns['JOB'])

    def test_preparation_consumes_generation_budget(self):
        clock=[100]
        def prepare(*args,**kwargs): clock[0]+=4; return ''
        self.ns.update(job_state=MagicMock(),prepare=prepare,chat_pause=MagicMock(return_value=False),
                       run_job=MagicMock(return_value=(True,['fixture.glb'],'')),release=MagicMock(),chat_resume=MagicMock())
        with patch.object(time,'monotonic',side_effect=lambda:clock[0]):
            self.ns['_run']({},10,10,lambda _:None,[],False,110)
        self.assertEqual(self.ns['run_job'].call_args.args[1],6)
        self.ns['release'].assert_called_once()

if __name__=='__main__':
    unittest.main()
