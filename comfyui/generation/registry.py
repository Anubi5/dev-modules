"""Reviewed family metadata and workflow recipes. Never import code from model data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]/'families'


def has_implementation(name):
    """Read-only discovery of a contained reviewed implementation, without executing it."""
    if not isinstance(name, str) or not re.fullmatch('[a-z0-9][a-z0-9-]*', name):
        return False
    trusted = Path(__file__).resolve().parents[1] / 'families'
    path = trusted / name / 'workflow.py'
    return not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(trusted.resolve())


def implementation(name):
    """Load reviewed code from a fixed source tree, never a metadata-supplied path or weights."""
    import importlib.util
    import sys
    if not isinstance(name, str) or not re.fullmatch('[a-z0-9][a-z0-9-]*', name):
        raise ValueError('Invalid reviewed builder identity')
    trusted = Path(__file__).resolve().parents[1] / 'families'
    path = trusted / name / 'workflow.py'
    if path.is_symlink() or not path.resolve().is_relative_to(trusted.resolve()) or not path.is_file():
        raise ValueError('No reviewed builder for architecture ' + name)
    key = 'eva_family_' + name.replace('-', '_')
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[key] = module
    return sys.modules[key]


def samples(prof):
    """Offline examples follow the selected reviewed implementation, never an external path."""
    import importlib.util
    owner = implementation((family(prof) or {}).get('builder', prof['arch']))
    path = Path(owner.__file__).with_name('samples.py')
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Escaped family sample file')
    spec = importlib.util.spec_from_file_location(owner.__name__ + '_samples', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assets():
    """Every asset models.txt declares: {(folder, file): (sha256, bytes or None)} - the one declaration per file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('fetch_models_declarations', ROOT.parent/'fetch_models.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return {(r['folder'], r['name']): (r['sha256'].lower(), r['bytes'])
            for r in module.declarations(ROOT.parent/'models.txt') if r['sha256']}


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
        declared = assets()
        for part in data['parts'].values():
            relative = PurePosixPath(part['folder'])/part['file']
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative):
                raise ValueError('Unsafe family model path: '+name)
            pin = declared.get((part['folder'], part['file']))
            if pin is not None:            # declared once, in models.txt: the package names the file only
                if 'sha256' in part or 'bytes' in part:
                    raise ValueError(f'{part["folder"]}/{part["file"]} is declared in models.txt; {name} must not repeat its pin')
                part['sha256'], part['bytes'] = pin
            elif (not re.fullmatch('[a-f0-9]{64}', str(part.get('sha256'))) or not isinstance(part.get('bytes'), int)
                    or part['bytes'] <= 0):
                raise ValueError('Family parts not in models.txt require a size and SHA-256: '+name)
            if not isinstance(part.get('optional', False), bool):
                raise ValueError('Family part "optional" must be true or false: '+name)
        if data['kind'] == 'video' and data.get('builder') is None:
            raise ValueError('A video family needs its reviewed builder: '+name)
        if data['kind'] == '3d':
            # A 3D family runs a reviewed recipe file shipped with the module (recipes/*.json), named by its profile.
            workflow = PurePosixPath(str(data.get('defaults', {}).get('workflow', '')))
            if (workflow.parts[:1] != ('recipes',) or len(workflow.parts) != 2 or workflow.suffix != '.json'
                    or not (ROOT.parent / workflow).is_file() or not (ROOT.parent / workflow).resolve().is_relative_to(ROOT.parent.resolve()) or 'builder' in data or 'recipe' in data):
                raise ValueError('A 3D family names one reviewed recipes/*.json file and nothing else: '+name)
            data['sha256'] = hashlib.sha256(raw).hexdigest()
            result[name] = data
            continue
        builder = data.get('builder')
        if builder is not None:
            # A reviewed graph builder shipped with the module (families/<builder>/workflow.py), named, never a path.
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
    if data is None:return
    root=Path(models_root).resolve()
    for part in data['parts'].values():
        if part.get('optional'):continue                     # used only by an option that checks for it itself
        path=root/'comfy'/part['folder']/part['file']
        if not path.resolve().is_relative_to(root) or not path.is_file() or (part.get('bytes') is not None and path.stat().st_size!=part['bytes']):
            raise ValueError(data['name']+' needs '+str(path)+' with its declared size (two-phase download plan or owner copy); copy the pinned component and run ai models verify '+data['name'])


def implementation_identity(prof):
    """Hash selected family data and the shipped local code it imports, without reading weights.

    Unrelated family files are excluded. Selected defaults, shared graph/features and
    the family's recipe participate; a repository-wide commit is deliberately not the key.
    """
    import ast
    data = family(prof)
    root = ROOT.parent
    selected = dict(prof)
    payload = dict(profile=selected, family=data)
    defaults = json.loads((root / 'profiles.json').read_text())
    payload['styles'] = defaults.get('styles', {})
    if data and data.get('builder'):
        extra = getattr(implementation(data['builder']), 'acceptance_defaults', None)
        if extra:
            payload['composition_defaults'] = extra()
    pending = []
    if data and data.get('builder'):
        pending.append(Path(implementation(data['builder']).__file__))
    elif not data and has_implementation(prof.get('arch')):
        pending.append(Path(implementation(prof['arch']).__file__))
    if data and data['kind'] == '3d':
        pending.append(root / data['defaults']['workflow'])
    pending.append(root / 'generation/graph.py')
    files = {}
    while pending:
        path = pending.pop().resolve()
        if path in files or not path.is_file():
            continue
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Acceptance implementation escapes reviewed module')
        raw = path.read_bytes()
        files[path] = hashlib.sha256(raw).hexdigest()
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = root.joinpath(*(node.module or '').split('.')) if not node.level else path.parent.joinpath(*(['..'] * (node.level - 1)), *(node.module or '').split('.'))
            for candidate in [base] + [base / alias.name for alias in node.names]:
                local = candidate.with_suffix('.py')
                if local.is_file():
                    pending.append(local)
    payload['code'] = {p.relative_to(root.resolve()).as_posix(): value for p, value in files.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
