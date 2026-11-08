"""Family packages are inert metadata; builders preserve existing graphs and ownership."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from generation import builders, image, models, registry
import check_workflows


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.prof=models.profile('krea2')
        self.values=dict(prompt='a ceramic teapot',negative='',seed=19,steps=8,width=1024,height=1024,prefix='test',pose_strength=.65)
        self.options=SimpleNamespace(pose=None,face=None,fix_faces=False,fix_hands=False,hires=False,
                                     upscale=False,face_weight=.65,image=[])

    def test_krea_package_builds_without_central_dispatch_or_other_family_changes(self):
        wf=builders.build(self.prof,self.values,self.options)
        self.assertEqual(wf['2']['inputs']['type'],'krea2')
        self.assertEqual(wf['4']['inputs']['text'],self.values['prompt'])
        self.assertEqual(wf['8']['inputs']['seed'],19)
        schema=json.loads(Path(__file__).with_name('schema.json').read_text())
        unknown=set(); self.assertEqual(check_workflows.check(wf,schema,'krea2',unknown),[])
        self.assertFalse(unknown)

    def test_removing_package_refuses_that_family_but_keeps_existing_builders(self):
        with patch.object(registry,'packages',return_value={}):
            with self.assertRaisesRegex(ValueError,'removed'): builders.build(self.prof,self.values,self.options)
            p=models.profile('zimage')
            self.assertEqual(builders.build(p,self.values,self.options),image.zimage(self.values['prompt'],19,8,1024,1024,'test',p))

    def test_discovery_does_not_import_a_package_python_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); p=root/'krea2'; p.mkdir()
            (p/'package.json').write_text(json.dumps(registry.family(self.prof)))
            (p/'__init__.py').write_text("raise AssertionError('arbitrary package code executed')")
            self.assertEqual(set(registry.packages(root)),{'krea2'})

    def test_invalid_paths_and_bindings_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); p=root/'krea2';p.mkdir(); package=deepcopy(registry.family(self.prof))
            for change in ['path','binding']:
                bad=deepcopy(package)
                if change=='path': bad['parts']['model']['file']='../../escape.safetensors'
                else: bad['recipe']['inputs']['seed']['set']=[['missing','seed']]
                (p/'package.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError): registry.packages(root)

    def test_existing_sdxl_advanced_graph_is_identical(self):
        o=self.options; o.pose='photo';o.face='face';o.fix_faces=True;o.fix_hands=True;o.hires=True;o.upscale=True
        p=models.profile('realvis');v=self.values
        expected=image.sdxl(v['prompt'],v['negative'],19,8,1024,1024,True,'test',pose='pending',face='pending',
            fix_faces=True,fix_hands=True,hires=True,pose_strength=.65,face_weight=.65,prof=p)
        self.assertEqual(builders.build(p,v,o),expected)

    def test_declarative_inputs_are_bounded_and_package_objects_are_detached(self):
        with self.assertRaises(SystemExit): builders.build(self.prof,dict(self.values,width=99999),self.options)
        first=registry.family(self.prof);first['parts']['model']['file']='changed'
        self.assertNotEqual(registry.family(self.prof)['parts']['model']['file'],'changed')

    def test_use_time_part_check_is_cheap_and_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'comfy/diffusion_models/model.safetensors';p.parent.mkdir(parents=True)
            p.write_bytes(b'data')
            data=dict(name='fixture',parts={'model':dict(folder='diffusion_models',file=p.name,bytes=4)})
            with patch.object(registry,'family',return_value=data),patch.object(Path,'open',side_effect=AssertionError('Weights opened at use-time')):
                registry.check_parts(self.prof,root)
            p.write_bytes(b'wrong size')
            with patch.object(registry,'family',return_value=data),self.assertRaisesRegex(ValueError,'declared size'):
                registry.check_parts(self.prof,root)


if __name__=='__main__': unittest.main()
