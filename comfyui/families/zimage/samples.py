"""Offline zimage graph examples."""
from families.zimage.workflow import zimage


def samples(p):
    """Offline graph regression fixture."""
    import itertools
    for up in (False, True):
        yield zimage('x', 1, p['steps'], 1024, 1024, 't', p, upscale=up)
        yield zimage('x', 1, p['steps'], 1024, 1024, 't', p, upscale=up, hires=True)


def lora_sample(p):
    """Offline graph regression fixture."""
    w = zimage('x', 1, p['steps'], 1024, 1024, 't', p)
    return w
