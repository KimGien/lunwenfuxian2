"""Resource bindings and isolated imports for the three author adapters."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

WORK = Path(__file__).resolve().parents[3]
MODULE = Path(__file__).resolve().parents[1]
ASSETS = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets'
DEPENDENCIES = ASSETS / 'python_dependencies'
_LOADED = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_resources(name, filename):
    data = json.loads((ASSETS / 'download_manifest.json').read_text(encoding='utf-8'))
    source = data['sources'][name]
    # Byte identity is checked for every Python file in the pinned author snapshot.
    for item in source['files']:
        if item['path'].endswith('.py'):
            require(digest(WORK / item['path']) == item['sha256'], 'Author source changed: ' + item['path'])
    weight = next(f for f in data['weights'][name]['files'] if Path(f['path']).name == filename)
    path = WORK / weight['path']
    require(path.is_file() and path.stat().st_size == weight['bytes'] and digest(path) == weight['sha256'],
            'Formal checkpoint missing or changed: ' + str(path))
    return path, {'source': {'repository': source['repository'], 'revision': source['revision'],
                            'archive_sha256': source['archive']['sha256'], 'source_modified': False},
                  'checkpoint': weight}


def enable_dependencies():
    # This target includes a Windows torchvision wheel. Linux Mamba uses its
    # separately created environment; never put Windows binary packages first.
    if sys.platform == 'win32' and str(DEPENDENCIES) not in sys.path:
        sys.path.insert(0, str(DEPENDENCIES))


def namespace(name, path):
    value = types.ModuleType(name)
    value.__path__ = [str(path)]
    value.__package__ = name
    sys.modules[name] = value


def author_module(name, package, module, torcheeg_constants=False, sibling_packages=()):
    """Import author code without retaining generic `models`/`Models` aliases.

    Model functions retain their original module globals. For LUNA, only the
    torcheeg constants package is needed, so its broad dataset loader is bypassed.
    No author code or constant is replaced.
    """
    key = (name, package, module)
    if key in _LOADED:
        return _LOADED[key]
    enable_dependencies()
    prefixes = [package, *sibling_packages] + (['torcheeg'] if torcheeg_constants else [])
    matching = lambda k: any(k == prefix or k.startswith(prefix + '.') for prefix in prefixes)
    previous = {k: v for k, v in list(sys.modules.items()) if matching(k)}
    for k in previous:
        del sys.modules[k]
    root = MODULE / 'vendor' / name
    namespace(package, root / package)
    for sibling in sibling_packages:
        namespace(sibling, root / sibling)
    if torcheeg_constants:
        namespace('torcheeg', DEPENDENCIES / 'torcheeg')
        namespace('torcheeg.datasets', DEPENDENCIES / 'torcheeg/datasets')
    try:
        loaded = importlib.import_module(package + '.' + module)
        _LOADED[key] = loaded
        return loaded
    finally:
        for k in list(sys.modules):
            if matching(k):
                del sys.modules[k]
        sys.modules.update(previous)


def raw_contract(raw):
    import numpy as np
    require(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.ndim == 3
            and raw.shape[1:] == (19, 1000) and len(raw) > 0 and bool(np.isfinite(raw).all()),
            'Expected finite unstandardized E2v2 float32 [windows,19,1000]')


def parameter_names(model, prefixes):
    names = [n for n, _ in model.named_parameters() if any(n == p or n.startswith(p + '.') for p in prefixes)]
    require(names, 'Registered unfreeze modules have no parameters')
    return names


def select_trainability(model, trainable, names):
    model.requires_grad_(False)
    allowed = set(names)
    if trainable:
        for name, parameter in model.named_parameters():
            if name in allowed:
                parameter.requires_grad_(True)
    model.eval()
    return model
