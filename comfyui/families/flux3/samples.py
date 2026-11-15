"""Offline flux3 graph examples."""
from families.flux3.workflow import flux3


def samples(p):
    """Offline graph regression fixture."""
    import itertools
    for refs, up in itertools.product([(), ('a',), ('a', 'b')], [False, True]):
        yield flux3('x', 1, p['steps'], 1024, 1024, 't', p, refs=refs, upscale=up)


def lora_sample(p):
    """Offline graph regression fixture."""
    w = flux3('x', 1, p['steps'], 1024, 1024, 't', p, refs=('a',))
    return w
