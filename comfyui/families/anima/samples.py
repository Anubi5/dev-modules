"""Offline anima graph examples."""
from families.anima.workflow import anima


def samples(p):
    """Offline graph regression fixture."""
    import itertools
    for pose, up in itertools.product([None, 'p'], [False, True]):
        yield anima('x', 'y', 1, p['steps'], 1024, 1024, 't', p, pose=pose, upscale=up)


def lora_sample(p):
    """Offline graph regression fixture."""
    w = anima('x', 'y', 1, p['steps'], 1024, 1024, 't', p, pose='p')
    return w
