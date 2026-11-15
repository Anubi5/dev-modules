"""Video dispatch through the same reviewed family loader as image generation."""
from generation.models import package
from generation import registry
from families.wan22.workflow import wan  # stable public graph API


def architectures():
    return frozenset(p['architecture'] for p in registry.packages().values() if p['kind'] == 'video')


ARCHITECTURES = architectures()


def build(prompt, negative, seed, steps, frames, prefix, model=None):
    selected = package(model, kind='video')
    implementation = registry.implementation(registry.family(selected.settings)['builder'])
    return implementation.build(selected, prompt, negative, seed, steps, frames, prefix)
