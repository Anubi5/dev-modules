"""Generic family capability and graph dispatch; metadata never names an executable path."""
from . import registry


def capabilities(prof):
    package = registry.family(prof)
    if package and not registry.moved(package):
        result = dict(features=[], uploads={}, negative=False, edit=False, expand=False, upscale=False, timeout=900)
        result.update(package.get('capabilities', {}))
        return result
    builder = package['builder'] if package else prof['arch']
    result = dict(features=[], uploads={}, negative=True, edit=False, expand=False, upscale=True, timeout=900)
    result.update(registry.implementation(builder).CAPABILITIES)
    if package:
        result.update(package.get('capabilities', {}))
    return result


def build(prof, values, options):
    if registry.declarative(prof):
        return registry.build(prof, values)
    return capabilities(prof)['build'](prof, values, options)
