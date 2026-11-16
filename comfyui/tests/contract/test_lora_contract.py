"""Family-owned loader declarations preserve wiring and refuse guessed graphs."""
import copy
import unittest

from features.lora import apply_loras
from generation import builders, models, registry


class LoraContractTests(unittest.TestCase):
    def test_declared_loader_connections_chain_without_touching_vae_or_refiner(self):
        for name in ('realvis','zimage','flux3','fibo2','anima','moss'):
            profile=models.profile(name)
            wf=registry.samples(profile).lora_sample(profile)
            original=copy.deepcopy(wf)
            spec=builders.capabilities(profile)['lora']
            apply_loras(wf,[('a.safetensors',0.8),('b.safetensors',-0.5)],profile['arch'])
            self.assertEqual(wf['60']['inputs']['model'],spec['model'])
            self.assertEqual(wf['60']['inputs']['clip'],spec['clip'])
            self.assertEqual(wf['61']['inputs']['model'],['60',0])
            self.assertEqual(wf['61']['inputs']['clip'],['60',1])
            for node, value in original.items():
                for key, link in value['inputs'].items():
                    expected=['61',0] if link==spec['model'] else ['61',1] if link==spec['clip'] else link
                    self.assertEqual(wf[node]['inputs'][key],expected,(name,node,key))

    def test_unknown_architecture_and_wrong_loader_are_refused(self):
        with self.assertRaisesRegex(ValueError,'not declared'):
            apply_loras({},[('a',1)],'krea2')
        wf={'1':{'class_type':'UnsupportedQuantizedLoader','inputs':{}},'2':{'class_type':'CLIPLoader','inputs':{}}}
        before=copy.deepcopy(wf)
        with self.assertRaisesRegex(ValueError,'incompatible'):
            apply_loras(wf,[('a',1)],'flux3')
        self.assertEqual(wf,before)

    def test_variants_inherit_the_architecture_capabilities(self):
        for variants in (('realvis','juggernaut','sdxl-base'),('anima','anima-turbo')):
            owner=builders.capabilities(models.profile(variants[0]))
            for name in variants[1:]:
                capability=builders.capabilities(models.profile(name))
                for key in ('features','lora','negative','edit'):
                    self.assertEqual(capability.get(key),owner.get(key),(name,key))


if __name__ == '__main__':
    unittest.main()
