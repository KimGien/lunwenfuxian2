"""EEG2Rep precision repair: BF16 encoder, strictly FP32 complete SSL loss.

All sampling, SSP, logical batch, validation, optimizer, EMA, checkpoint and
recovery control flow delegates to the immutable original ssl.py. Only the
loss autocast boundary and truthful precision provenance are changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch

from . import ssl as V1, storage as S
from .adapters import eeg2rep as E

SOURCE = Path(__file__).resolve()
_ORIGINAL_LOSS = E.ssl_loss
_ORIGINAL_SAVE = S.save
COMPONENTS = ('loss', 'alignment_mse', 'variance_hinge', 'offdiagonal_covariance', 'prediction_std_mean')


def precision_provenance():
    return {'version': 2, 'wrapper_source': S.desc(SOURCE), 'delegated_runner_source': S.desc(Path(V1.__file__)),
            'unchanged_adapter_source': S.desc(Path(E.__file__)),
            'repair': 'explicit autocast disabled only during complete SSL loss; .float() alone does not prevent autocast matmul',
            'encoder_precision': 'original registered float32 or bfloat16', 'loss_internal_statistics_precision': 'float32',
            'covariance_matmul_precision': 'float32', 'same_ssp_logical_batch_ema_optimizer_stop_and_recovery': True,
            'not_selected_using_validation_or_test_performance': True}


def ssl_loss(output):
    prediction = output['prediction_masked']
    # Retain the original differentiable cast and detached target; autocast is
    # restored immediately on leaving this loss call, including exceptions.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        value = _ORIGINAL_LOSS(output)
    if any(value[name].dtype != torch.float32 for name in COMPONENTS):
        raise ValueError('SSL v2 full loss/statistics must stay float32')
    return value


def augment_record(kind, value):
    """Add provenance before the immutable settings/contract/metadata write."""
    repair = precision_provenance()
    value['ssl_precision_repair'] = repair
    if kind == 'settings':
        value['loss_precision'] = 'float32'
        value['loss_autocast_enabled'] = False
        sources = value['prepared_sources']
        for record in (repair['wrapper_source'], repair['delegated_runner_source'], repair['unchanged_adapter_source']):
            if record not in sources:
                sources.append(record)
    elif kind == 'metadata':
        value['loss_precision'] = 'float32'
        value['loss_autocast_enabled'] = False
    elif kind == 'contract':
        value['input_contract']['ssl_precision_repair'] = repair
        if repair['wrapper_source'] not in value['implementation_sources']:
            value['implementation_sources'].append(repair['wrapper_source'])
    return value


def _save_with_provenance(path, value, *, immutable=False):
    path = Path(path).resolve()
    if path == V1.SETTINGS.resolve():
        augment_record('settings', value)
    elif path == V1.ENGINEERING_CONTRACT.resolve():
        augment_record('contract', value)
    elif path.name == 'metadata.json' and path.is_relative_to((S.OUT / 'runs' / V1.CONDITION).resolve()):
        augment_record('metadata', value)
    elif path.is_relative_to((S.OUT / 'ssl/engineering_attempts').resolve()):
        value['ssl_precision_repair'] = precision_provenance()
    return _ORIGINAL_SAVE(path, value, immutable=immutable)


def _verify_existing_precision_settings():
    if V1.SETTINGS.exists():
        saved = S.read(V1.SETTINGS)
        if saved.get('ssl_precision_repair') != precision_provenance() or saved.get('loss_autocast_enabled') is not False:
            raise ValueError('Existing SSL settings lack this v2 precision lock; refuse silent adoption or overwrite')
        for record in saved['prepared_sources']:
            S.checked(record)


@contextmanager
def activate():
    """All three original prepare/fit/validation paths resolve E.ssl_loss here."""
    _verify_existing_precision_settings()
    previous_loss, previous_save = E.ssl_loss, S.save
    E.ssl_loss = ssl_loss
    S.save = _save_with_provenance
    try:
        yield
    finally:
        E.ssl_loss = previous_loss
        S.save = previous_save


def prepare(*, logical_batch=64, precision='bfloat16'):
    with activate():
        return V1.prepare(logical_batch=logical_batch, precision=precision)


def run(*, seed=None, fold=None, resume=False):
    with activate():
        return V1.run(seed=seed, fold=fold, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--logical-batch', type=int, default=64)
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--seed', type=int, choices=V1.P.SEEDS)
    parser.add_argument('--fold', type=int, choices=V1.P.FOLDS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(logical_batch=args.logical_batch, precision=args.precision)
    else:
        run(seed=args.seed, fold=args.fold, resume=args.resume)


if __name__ == '__main__':
    main()
