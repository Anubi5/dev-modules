"""Reviewed family metadata and workflow recipes. Never import code from model data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]/'families'


def packages(root=ROOT):
    result = {}
    if not root.exists(): return result
    for path in sorted(root.glob('*/package.json')):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Family metadata must stay in the reviewed source root')
        raw = path.read_bytes(); data = json.loads(raw)
        name = data.get('name', '')
        if data.get('schema') != 'local-ai.family/1' or not re.fullmatch('[a-z0-9][a-z0-9.-]*', name):
            raise ValueError('Invalid family package: '+str(path))
        if name in result: raise ValueError('Duplicate family package: '+name)
        if data.get('kind') not in ('image', 'video', '3d') or not data.get('architecture') or not data.get('parts'):
            raise ValueError('Incomplete family: '+name)
        if data['kind'] == 'video' and data.get('builder') is None:
            raise ValueError('A video family needs its reviewed builder: '+name)
        if data['kind'] == '3d':
            # A 3D family runs a reviewed recipe file shipped with the module (recipes/*.json), named by its profile.
            workflow = PurePosixPath(str(data.get('defaults', {}).get('workflow', '')))
            if (workflow.parts[:1] != ('recipes',) or len(workflow.parts) != 2 or workflow.suffix != '.json'
                    or not (ROOT.parent / workflow).is_file() or 'builder' in data or 'recipe' in data):
                raise ValueError('A 3D family names one reviewed recipes/*.json file and nothing else: '+name)
            data['sha256'] = hashlib.sha256(raw).hexdigest()
            result[name] = data
            continue
        for part in data['parts'].values():
            relative = PurePosixPath(part['folder'])/part['file']
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative):
                raise ValueError('Unsafe family model path: '+name)
            if not re.fullmatch('[a-f0-9]{64}', part['sha256']) or part['bytes'] <= 0:
                raise ValueError('Family parts require a size and SHA-256: '+name)
            if not isinstance(part.get('optional', False), bool):
                raise ValueError('Family part "optional" must be true or false: '+name)
        builder = data.get('builder')
        if builder is not None:
            # A reviewed graph builder shipped with the module (generation/builders.BUILTINS), named, never a path.
            if not isinstance(builder, str) or not re.fullmatch('[a-z0-9][a-z0-9-]*', builder) or 'recipe' in data:
                raise ValueError('Family builder must be a reviewed builder name (and no recipe): '+name)
            data['sha256'] = hashlib.sha256(raw).hexdigest()
            result[name] = data
            continue
        recipe = data.get('recipe', {})
        workflow = recipe.get('workflow', {})
        if not workflow or not recipe.get('inputs'): raise ValueError('Family needs a workflow recipe: '+name)
        for spec in recipe['inputs'].values():
            for node, key in spec['set']:
                if node not in workflow or key not in workflow[node]['inputs']:
                    raise ValueError('Family binding targets an absent input: '+name)
        data['sha256'] = hashlib.sha256(raw).hexdigest()
        result[name] = data
    return result


def profiles():
    result = {}
    for name, data in packages().items():
        p = deepcopy(data['defaults'])
        p.update(arch=data['architecture'], kind=data['kind'], family=name)
        p.update(deepcopy(data.get('catalogue', {})))
        if 'description' in data or data['kind'] == 'image':
            p['about'] = data.get('description', name)
        for role, part in data['parts'].items():
            p['file' if role == 'model' else role] = part['file']
            if part.get('optional'):
                p[role + '_optional'] = True
        result[name] = p
    return result


def declarative(prof):
    """A family made only of its package's recipe (not a moved profile: see moved())."""
    data = family(prof)
    return bool(data) and not moved(data)


def moved(data):
    """A profile moved into a family package (3.2): it names a reviewed builder, or a reviewed 3D recipe file."""
    return bool(data.get('builder')) or data.get('kind') == '3d'


def family(prof):
    name = prof.get('family')
    if not name: return None
    data = packages().get(name)
    if data is None: raise ValueError('Family package was removed: '+name)
    return data


def build(prof, values):
    from .graph import apply_recipe
    data = family(prof)
    if data is None: raise ValueError('Expected a declarative family')
    parameters = dict(values)
    for role, part in data['parts'].items():
        parameters[role] = prof['file' if role == 'model' else role]
    return apply_recipe(data['recipe'], parameters)


def check_parts(prof, models_root):
    """Cheap use-time check; full hashes belong to explicit verification."""
    data=family(prof)
    if data is None or moved(data):return            # moved profiles: ComfyUI's preflight, as before
    root=Path(models_root).resolve()
    for part in data['parts'].values():
        if part.get('optional'):continue                     # used only by an option that checks for it itself
        path=root/'comfy'/part['folder']/part['file']
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size!=part['bytes']:
            raise ValueError(data['name']+' needs '+str(path)+' with its declared size; copy the pinned component and run ai models verify '+data['name'])
