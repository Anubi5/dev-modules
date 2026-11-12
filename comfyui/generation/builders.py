"""Narrow adapters for existing graph APIs; new declarative families need no dispatch edits."""
from . import image, registry


def _sdxl(p, v, o):
    return image.sdxl(v['prompt'], v['negative'], v['seed'], v['steps'], v['width'], v['height'], o.upscale,
        v['prefix'], pose='pending' if o.pose else None, face='pending' if o.face else None,
        fix_faces=o.fix_faces, fix_hands=o.fix_hands, hires=o.hires, pose_strength=v['pose_strength'],
        face_weight=o.face_weight, prof=p)


def _anima(p, v, o):
    return image.anima(v['prompt'], v['negative'], v['seed'], v['steps'], v['width'], v['height'],
        v['prefix'], p, pose='pending' if o.pose else None, pose_strength=v['pose_strength'], upscale=o.upscale)


def _zimage(p, v, o):
    return image.zimage(v['prompt'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p, upscale=o.upscale, hires=o.hires)


def _moss(p, v, o):
    return image.moss(v['prompt'], v['negative'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p, upscale=o.upscale)


def _flux3(p, v, o):
    return image.flux3(v['prompt'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p,
        refs=['pending']*len(o.image), upscale=o.upscale)


def _fibo2(p, v, o):
    return image.fibo2(v['prompt'], v['seed'], v['steps'], v['width'], v['height'], v['prefix'], p,
        refs=['pending']*len(o.image), upscale=o.upscale)


# These are trusted adapters shipped with the module, never executable paths from metadata or weights.
BUILTINS = {
    'sdxl': dict(build=_sdxl, features=['pose','face','fix_faces','fix_hands','hires'], uploads={'pose':'30','face':'21'}),
    'anima': dict(build=_anima, features=['pose'], uploads={'pose':'30'}),
    'zimage': dict(build=_zimage, negative=False, features=['hires']),
    'moss': dict(build=_moss),
    'flux3': dict(build=_flux3, negative=False, edit=True, expand=True),
    'fibo2': dict(build=_fibo2, negative=False, edit=True),
}


def capabilities(prof):
    package = registry.family(prof)
    if package:
        result = dict(features=[], uploads={}, negative=False, edit=False, expand=False, upscale=False)
        result.update(package.get('capabilities', {}))
        return result
    if prof['arch'] not in BUILTINS: raise ValueError('No reviewed builder for architecture '+prof['arch'])
    result = dict(features=[], uploads={}, negative=True, edit=False, expand=False, upscale=True)
    result.update(BUILTINS[prof['arch']])
    return result


def build(prof, values, options):
    if registry.family(prof): return registry.build(prof, values)
    return capabilities(prof)['build'](prof, values, options)
