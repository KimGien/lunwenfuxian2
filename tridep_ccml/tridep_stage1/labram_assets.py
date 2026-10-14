"""Pinned official LaBraM-base assets and strictly frozen patch-token extraction.

The author assumes microvolts at 200 Hz and divides this by 100 before forming
200-sample patches. Dataset physical calibration must be documented upstream;
MODMA uses the EEGLAB reading convention with calibration unconfirmed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import types
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
VENDOR = ROOT / 'vendor'
MANIFEST = ASSETS / 'labram_asset_manifest.json'
CHECKPOINT = ASSETS / 'labram-base.pth'
AUTHOR_REPO = '935963004/LaBraM'
AUTHOR_REVISION = 'c431221e6cfd23dbfa9950e0180682fb322b0548'
AUTHOR_FILES = ('modeling_finetune.py', 'modeling_pretrain.py', 'utils.py',
                'engine_for_finetuning.py', 'run_class_finetuning.py',
                'README.md', 'LICENSE', 'requirements.txt')
TIMM_REPO = 'huggingface/pytorch-image-models'
TIMM_TAG = 'v0.4.12'
TIMM_FILES = ('timm/models/layers/drop.py', 'timm/models/layers/helpers.py',
              'timm/models/layers/weight_init.py', 'LICENSE')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'TriDep-frozen-LaBraM-stage1'}), timeout=60)


def fetch(url, destination, upstream_git_blob=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.part')
        with request(url) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    if upstream_git_blob:
        size = destination.stat().st_size
        git_blob = hashlib.sha1(f'blob {size}\0'.encode('ascii'))
        with destination.open('rb') as handle:
            while chunk := handle.read(1024 * 1024):
                git_blob.update(chunk)
        require(git_blob.hexdigest() == upstream_git_blob, 'Official Git blob mismatch: ' + str(destination))
    return {'path': destination.relative_to(ROOT).as_posix(), 'url': url,
            'sha256': sha256(destination), 'bytes': destination.stat().st_size,
            'upstream_git_blob_sha1': upstream_git_blob}


def download_official_assets():
    """Download revision-pinned sources and official base weight to this workspace."""
    if MANIFEST.exists():
        return verify_assets()
    tree_url = f'https://api.github.com/repos/{AUTHOR_REPO}/git/trees/{AUTHOR_REVISION}?recursive=1'
    with request(tree_url) as response:
        tree = json.load(response)
    require(not tree.get('truncated'), 'Official source tree response is truncated')
    blobs = {item['path']: item['sha'] for item in tree['tree'] if item['type'] == 'blob'}
    with request(f'https://api.github.com/repos/{TIMM_REPO}/commits/{TIMM_TAG}') as response:
        timm_revision = json.load(response)['sha']
    records = []
    for name in AUTHOR_FILES:
        url = f'https://raw.githubusercontent.com/{AUTHOR_REPO}/{AUTHOR_REVISION}/{name}'
        records.append(fetch(url, VENDOR / 'labram' / name, blobs[name]))
    checkpoint_name = 'checkpoints/labram-base.pth'
    checkpoint_url = f'https://raw.githubusercontent.com/{AUTHOR_REPO}/{AUTHOR_REVISION}/{checkpoint_name}'
    records.append(fetch(checkpoint_url, CHECKPOINT, blobs[checkpoint_name]))
    for name in TIMM_FILES:
        url = f'https://raw.githubusercontent.com/{TIMM_REPO}/{timm_revision}/{name}'
        records.append(fetch(url, VENDOR / 'timm_helpers' / Path(name).name))
    result = {'official_repository': f'https://github.com/{AUTHOR_REPO}',
              'official_revision': AUTHOR_REVISION, 'official_tree_url': tree_url,
              'official_checkpoint_name': 'labram-base.pth',
              'timm_helper_repository': f'https://github.com/{TIMM_REPO}',
              'timm_helper_tag': TIMM_TAG, 'timm_helper_revision': timm_revision,
              'downloaded_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'files': records,
              'author_input_contract': {'physical_unit': 'microvolts', 'sampling_rate_hz': 200,
                                        'amplitude_divisor': 100, 'patch_samples': 200,
                                        'embedding_dim': 200, 'max_patches_per_channel': 16,
                                        'channel_index': 'official utils.standard_1020.index(name) + 1; CLS=0'},
              'runtime_import_adapter': 'Exact official modeling_finetune.py AST; replace only timm helper imports with pinned official helper modules and unused model-registration decorator with identity. No architecture/forward code edits.',
              'physical_scale_selection': 'No subject labels, validation/test scores or data normalization choose unit scale.'}
    json_write(MANIFEST, result)
    return verify_assets()


def verify_assets():
    manifest = json_read(MANIFEST)
    require(manifest['official_revision'] == AUTHOR_REVISION, 'Unexpected official LaBraM revision')
    for item in manifest['files']:
        path = ROOT / item['path']
        require(path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'], 'LaBraM asset lock mismatch: ' + item['path'])
    return manifest


def _load_helper(filename):
    path = VENDOR / 'timm_helpers' / filename
    namespace = {'__name__': '_tridep_official_timm_' + path.stem, '__file__': str(path)}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    return namespace


def official_channel_names():
    """Read the author's channel vocabulary without importing unrelated utils deps."""
    path = VENDOR / 'labram/utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'standard_1020' for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Author standard_1020 channel vocabulary missing')


def channel_indices(channels):
    vocabulary = official_channel_names()
    names = [name.upper() for name in channels]
    require(len(names) == len(set(names)), 'Repeated input EEG channel')
    require(all(name in vocabulary for name in names), 'EEG channel absent from author vocabulary')
    return [0] + [vocabulary.index(name) + 1 for name in names]


def _author_module():
    path = VENDOR / 'labram/modeling_finetune.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    replaced = []
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in ('timm.models.layers', 'timm.models.registry'):
            replaced.append(node.module)
        else:
            body.append(node)
    require(sorted(replaced) == ['timm.models.layers', 'timm.models.registry'], 'Unexpected author dependency adapter surface')
    tree.body = body
    module = types.ModuleType('_tridep_labram_author_finetune')
    module.__file__ = str(path)
    module.__dict__.update({'drop_path': _load_helper('drop.py')['drop_path'],
                           'to_2tuple': _load_helper('helpers.py')['to_2tuple'],
                           'trunc_normal_': _load_helper('weight_init.py')['trunc_normal_'],
                           'register_model': lambda function: function})
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def load_frozen_labram(device='cpu'):
    """Load all official base encoder tensors, then freeze and return model + audit."""
    manifest = verify_assets()
    # The pinned official training checkpoint has argparse/NumPy scalar metadata.
    # Keep weights_only=True and allow only its known, non-executable data types.
    with torch.serialization.safe_globals([argparse.Namespace, np.dtype, np.dtypes.Float64DType,
                                           (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar')]):
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    require('model' in checkpoint and isinstance(checkpoint['model'], dict), 'Official model state dictionary missing')
    original_state = checkpoint['model']
    author = _author_module()
    model = author.labram_base_patch200_200(num_classes=0, EEG_size=1600,
                                         qkv_bias=False, use_abs_pos_emb=True,
                                         use_rel_pos_bias=False, use_mean_pooling=False,
                                         init_values=0.1, drop_rate=0.0,
                                         attn_drop_rate=0.0, drop_path_rate=0.0)
    target = model.state_dict()
    kept = {}
    mapping = {}
    excluded = []
    for key, value in original_state.items():
        # Author finetuning explicitly transfers student.* and drops the pretraining head.
        if not key.startswith('student.'):
            excluded.append({'key': key, 'reason': 'outside official student encoder'})
            continue
        candidate = key[len('student.'):]
        if candidate not in target:
            excluded.append({'key': key, 'reason': 'pretraining-only tensor absent from downstream encoder'})
            continue
        require(tuple(value.shape) == tuple(target[candidate].shape), 'LaBraM tensor shape mismatch: ' + key)
        kept[candidate] = value
        mapping[candidate] = key
    require(set(kept) == set(target), 'Missing official encoder tensors: ' + ', '.join(sorted(set(target) - set(kept))))
    model.load_state_dict(kept, strict=True)
    require(all(torch.equal(model.state_dict()[key], value) for key, value in kept.items()), 'Loaded state differs from official checkpoint')
    model.requires_grad_(False)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model.to(device=device, dtype=torch.float32)
    audit = {'passed': True, 'official_revision': AUTHOR_REVISION,
             'checkpoint_sha256': sha256(CHECKPOINT), 'checkpoint_state_key': 'model',
             'checkpoint_encoder_prefix': 'student.', 'original_checkpoint_state_keys': len(original_state),
             'token_normalization': 'use_mean_pooling=False retains pretrained student.norm; patch tokens are averaged downstream, with no new fc_norm parameters.',
             'kept_encoder_keys': len(kept), 'target_encoder_keys': len(target),
             'strict_encoder_coverage': 1.0, 'missing_encoder_keys': [],
             'encoder_key_mapping': mapping, 'excluded_checkpoint_keys': excluded,
             'model_parameters': sum(p.numel() for p in model.parameters()),
             'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'eval_mode': not model.training, 'device': str(device),
             'input_unit_assumption': 'Native EEGLAB numeric values treated as microvolts; author fixed divisor 100. No unit selection by scores.',
             'physical_calibration_unconfirmed': True,
             'floating_point': {'dtype': 'float32', 'AMP': False,
                                'cuda_matmul_TF32': False, 'cudnn_TF32': True},
             'asset_manifest_sha256': sha256(MANIFEST)}
    return model, audit


@torch.inference_mode()
def extract_patch_tokens(model, eeg_microvolts, channels, batch_size=2, *, return_all_tokens=False):
    """Return float32 (epochs, channels * patches, 200), or +1 CLS token.

    eeg_microvolts must already have physical units established upstream and be
    sampled at 200 Hz, with shape (epochs, channels, samples). No standardizing,
    unit inference, learned pooling, or label-driven preprocessing is performed.
    """
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'LaBraM must stay frozen in eval mode')
    input_chans = channel_indices(channels)
    values = np.asarray(eeg_microvolts, dtype=np.float32)
    require(values.ndim == 3 and values.shape[0] > 0 and values.shape[1] == len(channels), 'Expected (epochs, channels, samples) physical EEG')
    require(values.shape[2] % 200 == 0 and 1 <= values.shape[2] // 200 <= 16, 'LaBraM requires 1..16 full 200-sample patches')
    require(np.isfinite(values).all(), 'Nonfinite EEG input')
    require(batch_size > 0, 'batch_size must be positive')
    n_patches = values.shape[2] // 200
    expected_tokens = len(channels) * n_patches + int(return_all_tokens)
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(values), batch_size):
        physical = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
        patches = (physical / 100.0).reshape(len(physical), len(channels), n_patches, 200)
        with torch.autocast(device_type=device.type, enabled=False):
            features = model.forward_features(patches, input_chans=input_chans,
                                              return_patch_tokens=not return_all_tokens,
                                              return_all_tokens=return_all_tokens)
        require(features.shape == (len(physical), expected_tokens, 200) and torch.isfinite(features).all(), 'Unexpected LaBraM representation')
        outputs.append(features.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def smoke(device='cpu'):
    """Synthetic shape/interface/state probe only; no data/label/model selection."""
    model, audit = load_frozen_labram(device)
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    generator = np.random.default_rng(20261005)
    eeg = generator.normal(0.0, 10.0, (2, len(channels), 800)).astype(np.float32)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    patches = extract_patch_tokens(model, eeg, channels)
    repeated = extract_patch_tokens(model, eeg, channels)
    all_tokens = extract_patch_tokens(model, eeg, channels, return_all_tokens=True)
    require(np.array_equal(patches, repeated), 'LaBraM repeated frozen probe changed')
    require(np.array_equal(patches, all_tokens[:, 1:]), 'Author patch/all-token interfaces disagree')
    require(all(torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items()), 'LaBraM probe changed state')
    audit.update({'probe': 'synthetic independent of all cohort data/labels',
                  'probe_input_shape': list(eeg.shape), 'patch_token_shape': list(patches.shape),
                  'all_token_shape': list(all_tokens.shape), 'channel_indices': channel_indices(channels),
                  'repeat_input_equal': True, 'state_unchanged': True,
                  'patch_equals_all_without_CLS': True})
    json_write(ASSETS / 'labram_load_and_shape_audit.json', audit)
    return audit


def probe_e2v2(device='cuda'):
    """Two already-frozen E2v2 windows; records existing cache hash and interface."""
    workspace = ROOT.parents[1]
    input_manifest = workspace / 'tridep_ccml/experiments/EEG-E2v2-37-fulltext/input_manifest.json'
    first = json_read(input_manifest)['records'][0]
    cache = workspace / first['new_cache']['path']
    require(sha256(cache) == first['new_cache']['sha256'], 'E2v2 probe cache SHA mismatch')
    raw = np.load(cache, allow_pickle=False)
    require(raw.shape == (60, 19, 1000) and raw.dtype == np.float32, 'Expected corrected E2v2 raw input')
    channels = ['FP2', 'FP1', 'F7', 'F3', 'F4', 'F8', 'FCz', 'C3', 'C4', 'T3',
                'CPz', 'T4', 'P3', 'Pz', 'P4', 'T5', 'T6', 'O1', 'O2']
    model, audit = load_frozen_labram(device)
    tokens = extract_patch_tokens(model, raw[:2], channels)
    require(tokens.shape == (2, 95, 200), 'E2v2 probe should have 19 channels x 5 patches')
    pooled = tokens.mean(axis=1, dtype=np.float32)
    require(pooled.shape == (2, 200) and np.isfinite(pooled).all(), 'E2v2 mean token probe failed')
    audit.update({'probe': 'first 2 existing E2v2 windows; no labels loaded or score-based selection',
                  'subject_id': first['subject_id'], 'input_manifest_sha256': sha256(input_manifest),
                  'input_cache': first['new_cache'], 'input_shape': list(raw[:2].shape),
                  'channels': channels, 'channel_indices': channel_indices(channels),
                  'patch_token_shape': list(tokens.shape), 'mean_pooled_shape': list(pooled.shape),
                  'pooling': 'arithmetic mean of 19 x 5 pretrained normalized patch tokens per 5-second window',
                  'unit_assumption': 'EEGLAB native numeric treated as microvolts; no SET units/calibration/gain metadata available',
                  'physical_calibration_unconfirmed': True,
                  'module_sha256': sha256(Path(__file__)),
                  'representation_is_official_pretrained': True})
    json_write(ASSETS / 'labram_e2v2_shape_audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--probe-e2v2', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.download:
        value = download_official_assets()
        print(json.dumps({'downloaded': True, 'files': len(value['files']), 'revision': AUTHOR_REVISION}))
    if args.smoke:
        value = smoke(args.device)
        print(json.dumps({key: value[key] for key in ('passed', 'kept_encoder_keys', 'strict_encoder_coverage', 'patch_token_shape', 'all_token_shape', 'trainable_parameters')}))
    if args.probe_e2v2:
        value = probe_e2v2(args.device)
        print(json.dumps({key: value[key] for key in ('passed', 'patch_token_shape', 'mean_pooled_shape', 'physical_calibration_unconfirmed')}))
    if not args.download and not args.smoke and not args.probe_e2v2:
        print(json.dumps({'verified': True, 'files': len(verify_assets()['files']), 'revision': AUTHOR_REVISION}))
