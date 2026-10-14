"""Pinned author EEGPT encoder and a fixed, label-free 512-to-200 interface.

Asset qualification constructs/loads the encoder but never calls its forward.
The vendor source remains byte-identical to the pinned official source.
"""
from __future__ import annotations

import ast
from collections import OrderedDict, defaultdict
from functools import partial
import hashlib
import importlib.util
import os
import pickle
import pickletools
from pathlib import Path
import types
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np

WORK = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vendor/EEGPT_mcae.py'
AUTHOR_REVISION = 'ea7282f698c406cd423ccbedbe9960853761144c'
SOURCE_URL = f'https://raw.githubusercontent.com/BINE022/EEGPT/{AUTHOR_REVISION}/downstream/Modules/models/EEGPT_mcae.py'
SOURCE_SHA256 = 'cfe3d786e6c351e9678f9f7183cd17dab0dfbb131590ef88e3e4a9efae728a95'
CHECKPOINT = WORK / 'work/models/eegpt_author_bound_20261006/eegpt_mcae_58chs_4s_large4E.ckpt'
CHECKPOINT_BYTES = 1020602524
CHECKPOINT_SHA256 = '9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0'
CHECKPOINT_AUTHOR_MD5 = 'a165305f329f51b57691c8a3131f9b79'
CHECKPOINT_PREFIX = 'target_encoder.'
PROJECTION_SEED = 20261005
CHANNEL_NAMES = ('FP2', 'FP1', 'F7', 'F3', 'F4', 'F8', 'FCZ', 'C3', 'C4',
                 'T7', 'CPZ', 'T8', 'P3', 'PZ', 'P4', 'P7', 'P8', 'O1', 'O2')
CHANNEL_IDS = (2, 0, 7, 9, 13, 15, 20, 27, 31, 25, 38, 33, 45, 47, 49, 43, 51, 59, 61)
ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
ENCODER_CONFIGURATION = {
    'img_size': [19, 1024], 'patch_size': 64, 'patch_stride': None,
    'embed_dim': 512, 'embed_num': 4, 'depth': 8, 'num_heads': 8,
    'mlp_ratio': 4., 'qkv_bias': True, 'drop_rate': 0.,
    'attn_drop_rate': 0., 'drop_path_rate': 0., 'layer_norm_eps': 1e-6,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.is_file(), 'Missing workspace asset: ' + str(path))
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': file_digest(path)}


def prepare_source():
    """Read/verify the already downloaded pinned source; no network or models."""
    require(SOURCE.is_file() and file_digest(SOURCE) == SOURCE_SHA256, 'Pinned official EEGPT source missing/changed')
    ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
    return {**descriptor(SOURCE), 'url': SOURCE_URL, 'official_revision': AUTHOR_REVISION,
            'bytes': SOURCE.stat().st_size, 'source_modified': False}


def source_channel_dictionary():
    """Recover the author's literal channel list without importing torch/vendor."""
    prepare_source()
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CHANNEL_DICT' for t in n.targets))
    comprehension = assignment.value
    require(isinstance(comprehension, ast.DictComp), 'Author channel dictionary contract changed')
    call = comprehension.generators[0].iter
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'enumerate',
            'Author channel enumeration changed')
    names = ast.literal_eval(call.args[0])
    require(len(names) == len(set(names)) == 62, 'Author channel dictionary must contain62 unique names')
    return {name.upper(): index for index, name in enumerate(names)}


def canonical_channels(channels=CHANNEL_NAMES):
    names = tuple(ALIASES.get(str(c).upper().strip('.'), str(c).upper().strip('.')) for c in channels)
    require(names == CHANNEL_NAMES, 'EEGPT requires the registered19 native-order channel names')
    return names


def channel_ids(channels=CHANNEL_NAMES):
    names = canonical_channels(channels)
    dictionary = source_channel_dictionary()
    ids = tuple(dictionary[name] for name in names)
    require(ids == CHANNEL_IDS, 'Author channel IDs differ from registered mapping')
    return np.asarray(ids, dtype=np.int64)[None, :]


def projection():
    """PCG64 local RNG, CPU float64 reduced QR, positive R diagonal, scale1.

    The saved/used matrix is float32. Neither labels nor EEG values are inputs;
    global NumPy state is untouched. No learned or data-derived projection.
    """
    generator = np.random.Generator(np.random.PCG64(PROJECTION_SEED))
    normal = generator.standard_normal((512, 200), dtype=np.float64)
    q, r = np.linalg.qr(normal, mode='reduced')
    signs = np.where(np.diag(r) < 0., -1., 1.)
    q = np.ascontiguousarray(q * signs[None, :], dtype=np.float64)
    require(np.isfinite(q).all() and np.max(np.abs(q.T @ q - np.eye(200))) < 1e-12,
            'Fixed projection QR orthogonality failed')
    value = np.ascontiguousarray(q, dtype=np.float32)
    return value, {'seed': PROJECTION_SEED, 'rng': 'numpy.Generator(PCG64)',
        'generation_dtype': 'float64', 'used_dtype': 'float32', 'shape': [512, 200],
        'QR': 'numpy.linalg.qr reduced CPU', 'sign_convention': 'multiply each Q column by -1 iff corresponding R diagonal<0; zero chooses+1',
        'extra_scale': 1., 'float64_orthogonality_max_error': float(np.max(np.abs(q.T @ q - np.eye(200)))),
        'float64_array_sha256': hashlib.sha256(q.tobytes()).hexdigest(),
        'float32_array_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
        'label_or_data_inputs': [], 'trainable_parameters': 0}


def verify_checkpoint():
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == CHECKPOINT_BYTES,
            'Formal EEGPT checkpoint missing/incomplete')
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with CHECKPOINT.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block); md5.update(block)
    require(sha.hexdigest() == CHECKPOINT_SHA256 and md5.hexdigest() == CHECKPOINT_AUTHOR_MD5,
            'EEGPT file does not match pinned full mirror SHA and author File-info MD5')
    return {'path': CHECKPOINT.relative_to(WORK).as_posix(), 'sha256': sha.hexdigest(),
            'bytes': CHECKPOINT_BYTES, 'author_file_info_md5': md5.hexdigest(),
            'author_file_id': 46452745, 'full_formal_checkpoint': True}


def encoder_state_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode('utf-8')); h.update(str(value.dtype).encode('ascii'))
        h.update(str(tuple(value.shape)).encode('ascii')); h.update(value.numpy().tobytes())
    return h.hexdigest()


def author_module():
    prepare_source()
    spec = importlib.util.spec_from_file_location('_tridep_pinned_eegpt_mcae', SOURCE)
    require(spec is not None and spec.loader is not None, 'Cannot import pinned official EEGPT encoder')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load_author_checkpoint():
    """Load the full hash-bound author file with an exact nine-GLOBAL gate.

    This protocol2 pickle contains POP, unsupported by torch2.8's restricted
    weights-only parser. The replacement supports the original pickle format
    and restricts every find_class call to the observed fixed globals below.
    File bytes, SHA256 and author MD5 must all pass before deserialization.
    """
    binding = verify_checkpoint()
    import torch
    allowed = {
        ('__builtin__', 'dict'): dict, ('__builtin__', 'getattr'): getattr,
        ('collections', 'OrderedDict'): OrderedDict, ('collections', 'defaultdict'): defaultdict,
        ('torch', 'FloatStorage'): torch.FloatStorage,
        ('torch._utils', '_rebuild_parameter'): torch._utils._rebuild_parameter,
        ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
        ('torch.optim.adamw', 'AdamW'): torch.optim.AdamW,
        ('torch.optim.lr_scheduler', 'OneCycleLR'): torch.optim.lr_scheduler.OneCycleLR,
    }
    with zipfile.ZipFile(CHECKPOINT) as archive:
        names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        require(len(names) == 1, 'Author pickle archive layout differs')
        ops = list(pickletools.genops(archive.read(names[0])))
    globals_found = {tuple(value.split(' ', 1)) for opcode, value, _ in ops if opcode.name == 'GLOBAL'}
    require(ops[0][0].name == 'PROTO' and ops[0][1] == 2 and globals_found == set(allowed)
            and not any(opcode.name == 'STACK_GLOBAL' for opcode, _, _ in ops),
            'Author metadata protocol/GLOBAL contract differs')

    class RestrictedAuthorUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            require((module, name) in allowed, 'Forbidden author metadata GLOBAL: ' + module + '.' + name)
            return allowed[module, name]

    restricted = types.SimpleNamespace(__name__='tridep_author_bound_restricted_pickle',
                                       Unpickler=RestrictedAuthorUnpickler)
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, pickle_module=restricted)
    audit = {'checkpoint': binding, 'pickle_protocol': 2,
        'weights_only': False, 'deserializer': 'torch.load with exact9-GLOBAL restricted pickle.Unpickler',
        'allowed_globals': sorted(module + '.' + name for module, name in allowed),
        'unknown_globals_allowed': False,
        'POP_count': sum(opcode.name == 'POP' for opcode, _, _ in ops),
        'weights_only_parser_incompatibility': 'torch2.8 weights-only parser/get_unsafe_globals_in_checkpoint rejects author POP opcode48',
        'checkpoint_modified': False, 'file_gate_before_unpickle': ['bytes', 'sha256', 'author_md5']}
    return payload, audit


def load_frozen_eegpt(device='cpu'):
    """Load all target_encoder tensors exactly; no forward, predictor or head."""
    source_binding = prepare_source()
    payload, deserialization = load_author_checkpoint()
    checkpoint_binding = deserialization['checkpoint']
    import torch
    module = author_module()
    # Construction consumes only a local fork of RNG; every state tensor must
    # then be replaced by an exact author checkpoint tensor before returning.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = module.EEGTransformer(img_size=[19, 1024], patch_size=64, patch_stride=None,
            embed_dim=512, embed_num=4, depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    require(isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict), 'Author Lightning state_dict missing')
    original = payload['state_dict']
    target = model.state_dict()
    kept, mapping = {}, {}
    for key, value in original.items():
        if key.startswith(CHECKPOINT_PREFIX):
            name = key[len(CHECKPOINT_PREFIX):]
            require(name in target, 'Unexpected target_encoder tensor: ' + key)
            require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(target[name].shape)
                    and value.dtype == target[name].dtype and bool(torch.isfinite(value).all()),
                    'Author encoder tensor shape/dtype/finite mismatch: ' + key)
            kept[name], mapping[name] = value, key
    require(set(kept) == set(target), 'Missing author encoder keys: ' + ', '.join(sorted(set(target) - set(kept))))
    model.load_state_dict(kept, strict=True)
    require(all(torch.equal(model.state_dict()[key], value) for key, value in kept.items()),
            'Strict loaded encoder differs from formal checkpoint')
    require(np.array_equal(model.prepare_chan_ids(CHANNEL_NAMES).cpu().numpy(), channel_ids()), 'Model channel-position IDs differ')
    model.requires_grad_(False).eval()
    fingerprint = encoder_state_digest(model)
    model.to(device=device, dtype=torch.float32)
    require(encoder_state_digest(model) == fingerprint and not model.training and not any(p.requires_grad for p in model.parameters()),
            'Frozen encoder state changed on device transfer')
    audit = {'passed': True, 'checkpoint': checkpoint_binding, 'source': source_binding,
        'state_dict_key': 'state_dict', 'encoder_prefix': CHECKPOINT_PREFIX,
        'configuration': ENCODER_CONFIGURATION, 'encoder_keys': len(target), 'loaded_encoder_keys': len(kept),
        'missing_encoder_keys': [], 'unexpected_encoder_keys': [], 'strict_encoder_coverage': 1.,
        'key_mapping': mapping, 'excluded_other_state_dict_keys': len(original) - len(kept),
        'model_parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': 0,
        'encoder_state_sha256': fingerprint, 'eval': True, 'forward_calls_during_loading': 0,
        'channel_names': list(CHANNEL_NAMES), 'channel_ids': list(CHANNEL_IDS), 'device': str(device),
        'no_random_encoder_residual': True, 'no_predictor_reconstructor_or_classifier': True,
        'checkpoint_deserialization': deserialization}
    del payload, original, kept, target
    return model, audit
