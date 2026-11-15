"""Offline moss graph examples."""
from families.moss.workflow import moss


def samples(p):
    """Offline graph regression fixture."""
    import itertools
    for up in (False, True):
        yield moss('x', 'y', 1, p['steps'], 1024, 1024, 't', p, upscale=up)


def lora_sample(p):
    """Offline graph regression fixture."""
    w = moss('x', 'y', 1, p['steps'], 1024, 1024, 't', p)
    return w
