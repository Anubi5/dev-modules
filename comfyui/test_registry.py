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
            p=models.profile('moss')                     # a model with no family package keeps its builder
            self.assertEqual(builders.build(p,self.values,self.options),image.moss(self.values['prompt'],'',19,8,1024,1024,'test',p))

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

    # Profiles moved from profiles.json to families/<name> (Phase 3.2), exactly as they were there, with their
    # place in the catalogue (= baseline order) and builder.
    BEFORE = {
        "realvis": (0, {"arch": "sdxl", "file": "RealVisXL_V5.0_fp16.safetensors", "sampler": "dpmpp_sde", "scheduler": "karras",
                        "cfg": 4.5, "steps": 30, "about": "photoreal, flattering portraits; best with --face", "licence": "openrail++",
                        "source": "huggingface.co/SG161222/RealVisXL_V5.0 (card: DPM++ SDE Karras, 30+ steps)", "capability_tags": ["image"]}),
        "juggernaut": (1, {"arch": "sdxl", "file": "juggernautXL_ragnarok.safetensors", "sampler": "dpmpp_2m_sde", "scheduler": "karras",
                           "cfg": 4.0, "steps": 35, "negative": {"photo": "blurry, low quality, watermark, text, deformed hands"},
                           "about": "photoreal, natural documentary look, strong scenes", "licence": "see civitai.com/models/133005",
                           "source": "civitai.com/models/133005 (Ragnarok: DPM++ 2M SDE, 30-40 steps, CFG 3-6, start with little negative)",
                           "capability_tags": ["image"]}),
        "sdxl-base": (2, {"arch": "sdxl", "file": "sd_xl_base_1.0.safetensors", "sampler": "dpmpp_2m", "scheduler": "karras", "cfg": 6.0,
                          "steps": 30, "about": "original SDXL: general and artistic, weakest at photos", "licence": "openrail++",
                          "source": "huggingface.co/stabilityai/stable-diffusion-xl-base-1.0", "capability_tags": ["image"]}),
        "zimage": (4, {"arch": "zimage", "file": "z_image_turbo_int8_convrot.safetensors",
                       "text_encoder": "qwen_3_4b_fp8_mixed.safetensors", "vae": "ae.safetensors", "sampler": "res_multistep",
                       "scheduler": "simple", "cfg": 1.0, "steps": 8, "shift": 3.0, "about": "Z-Image Turbo: fast photoreal, 8 steps",
                       "licence": "Apache 2.0",
                       "source": "huggingface.co/Comfy-Org/Z-Image-Turbo + Comfy blueprint Text to Image (Z-Image-Turbo) (8 steps, CFG 1, res_multistep, shift 3)",
                       "capability_tags": ["image"], "pose_patch": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
                       "pose_strength": 1.0, "pose_patch_optional": True}),
    }

    def test_migrated_families_keep_their_profile_place_and_capabilities(self):
        for name, (place, before_profile) in self.BEFORE.items():
            with self.subTest(family=name):
                prof = models.profile(name)
                self.assertEqual({k: v for k, v in prof.items() if k not in ('name', 'family', 'kind')}, before_profile)
                self.assertEqual((prof['family'], prof['kind']), (name, 'image'))
                self.assertEqual(list(models.PROFILES['models']).index(name), place)
                self.assertFalse(registry.declarative(prof))
                before = dict(features=[], uploads={}, negative=True, edit=False, expand=False, upscale=True)
                before.update(builders.BUILTINS[prof['arch']])
                self.assertEqual(builders.capabilities(prof), before)
                with patch.object(registry, 'packages', return_value={}), self.assertRaisesRegex(ValueError, 'removed'):
                    registry.family(prof)

    def test_family_parts_match_their_models_txt_pins(self):
        from fetch_models import declarations
        pins = {(r['folder'], r['name']): (r['sha256'].lower(), r['bytes']) for r in declarations(Path(__file__).with_name('models.txt'))}
        for name, data in registry.packages().items():
            if not data.get('builder'): continue
            for role, part in data['parts'].items():
                with self.subTest(family=name, part=role):
                    sha, size = pins[(part['folder'], part['file'])]
                    self.assertEqual(sha, part['sha256'])
                    if size is not None: self.assertEqual(size, part['bytes'])   # one pin lists no size yet (3.4)

    def test_builder_family_never_names_a_path_or_has_a_recipe(self):
        data = json.loads(Path(__file__).with_name('families').joinpath('zimage/package.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p = root/'zimage'; p.mkdir()
            for bad in ('../x.py', 'families/zimage/family.py', 'Zimage'):
                (p/'package.json').write_text(json.dumps(dict(data, builder=bad)))
                with self.subTest(builder=bad), self.assertRaises(ValueError): registry.packages(root)
            (p/'package.json').write_text(json.dumps(dict(data, recipe={'workflow': {}, 'inputs': {}})))
            with self.assertRaises(ValueError): registry.packages(root)


if __name__=='__main__': unittest.main()
