"""Offline fibo2 graph examples."""
from families.fibo2.workflow import fibo2


def samples(p):
    """Offline graph regression fixture."""
    import itertools
    for refs, up in itertools.product([(), ('a',), ('a', 'b')], [False, True]):
        yield fibo2('x', 1, p['steps'], 1024, 1024, 't', p, refs=refs, upscale=up)


def lora_sample(p):
    """Offline LoRA regression fixture."""
    w = fibo2('x', 1, p['steps'], 1024, 1024, 't', p, refs=('a',))
    return w
