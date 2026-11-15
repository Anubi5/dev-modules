"""Offline sdxl graph examples."""
from families.sdxl.workflow import sdxl


def samples(p):
    """Offline graph regression fixture."""
    import itertools
    for pose, face, ff, fh, hires, up in itertools.product([None, 'p'], [None, 'f'], *[[False, True]] * 4):
        yield sdxl('x', 'y', 1, p['steps'], 1024, 1024, up, 't', pose=pose, face=face, fix_faces=ff, fix_hands=fh, hires=hires, prof=p)


def lora_sample(p):
    """Offline graph regression fixture."""
    w = sdxl('x', 'y', 1, p['steps'], 1024, 1024, True, 't', pose='p', face='f', fix_faces=True, hires=True, prof=p)
    return w
