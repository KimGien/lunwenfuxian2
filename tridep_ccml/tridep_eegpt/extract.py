"""Registered paired4-second frozen encoder export; never fit an encoder.

Each person's original60 five-second windows keep their order. The first8
windows are repeated as the same batch8, rather than comparing batch8 with1.
Extra arrays record already-computed inputs/tokens for zero-forward auditing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import WORK, OUT, FEATURE_BUDGET, read, save, desc, checked, registration, runtime, now, np, torch
from . import assets

BATCH_SIZE = 8
REPEAT_WINDOWS = 8
FEATURES = OUT / 'features'
MANIFEST = OUT / 'features_manifest.json'
LEDGER = OUT / 'feature_ledger.json'
INPUT_MATH = {
    'raw_shape': [60, 19, 1000], 'raw_dtype': 'float32', 'raw_frequency_hz': 200,
    'eegpt_resample': 'scipy.signal.resample_poly(raw.astype(float64),up32,down25,axis=-1,window=(kaiser,5.0),padtype=constant)',
    'eegpt_resampled_samples': 1280, 'eegpt_slice': 'first1024 after complete5s resampling',
    'eegpt_frequency_hz': 256, 'eegpt_amplitude': 'divide resampled float64 native numeric by1000 then castfloat32; reader-convention mV',
    'labram_slice': 'raw first800 at native200Hz',
    'labram_amplitude': 'numpy float32 division by100, then float32 device tensor',
    'labram_patch': [19, 4, 200], 'labram_pooling': 'numpy.mean(tokens,axis=1,dtype=float32), excludeCLS',
    'eegpt_pooling': 'numpy.mean(tokens,axis=(1,2),dtype=float32) over16times and4summarytokens',
    'projection': 'saved fixed float32 P; float64 accumulation(native512 @ P) then castfloat32,scale1; interface batches8 preserve repeat equality',
    'physical_calibration_confirmed': False,
    'reference': 'unchanged E2v2 Cz/common; author pretraining average-reference difference disclosed',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def paired_inputs(raw):
    """Pure numerical adaptation, with no learned/data-dependent settings."""
    from scipy.signal import resample_poly
    require(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.shape == (60, 19, 1000)
            and np.isfinite(raw).all(), 'Expected finite accepted float32 E2v2[60,19,1000]')
    upsampled = resample_poly(raw.astype(np.float64), up=32, down=25, axis=-1,
                             window=('kaiser', 5.), padtype='constant')
    require(upsampled.shape == (60, 19, 1280), 'True200-to256Hz conversion must retain1280samples/5s')
    eegpt = np.ascontiguousarray(upsampled[..., :1024] / 1000., dtype=np.float32)
    labram = np.ascontiguousarray(raw[..., :800] / np.float32(100.), dtype=np.float32)
    require(np.isfinite(eegpt).all() and np.isfinite(labram).all(), 'Nonfinite fixed matched inputs')
    return {'eegpt': eegpt, 'labram': labram}


def eegpt_interface(tokens, projection):
    require(tokens.ndim == 4 and tokens.shape[1:] == (16, 4, 512) and tokens.dtype == np.float32
            and np.isfinite(tokens).all(), 'Expected finite float32 EEGPT[B,16,4,512]')
    require(projection.shape == (512, 200) and projection.dtype == np.float32 and np.isfinite(projection).all(),
            'Fixed projection must be finite float32[512,200]')
    require(len(tokens) > 0, 'Empty EEGPT interface batch')
    native_blocks, feature_blocks = [], []
    for start in range(0, len(tokens), BATCH_SIZE):
        # Match the first8 interface arithmetic as well as the neural batch8;
        # do not compare a BLAS matrix60 product with a matrix8 repeat product.
        native = np.ascontiguousarray(tokens[start:start + BATCH_SIZE].mean(axis=(1, 2), dtype=np.float32))
        native_blocks.append(native)
        feature_blocks.append(np.ascontiguousarray(native.astype(np.float64) @ projection.astype(np.float64), dtype=np.float32))
    native = np.concatenate(native_blocks, axis=0)
    features = np.concatenate(feature_blocks, axis=0)
    require(np.isfinite(features).all(), 'Nonfinite projected EEGPT features')
    return native, features


def labram_interface(tokens):
    require(tokens.ndim == 3 and tokens.shape[1:] == (76, 200) and tokens.dtype == np.float32
            and np.isfinite(tokens).all(), 'Expected finite LaBraM[B,19x4,200] patchtokens withoutCLS')
    require(len(tokens) > 0, 'Empty LaBraM interface batch')
    return np.concatenate([np.ascontiguousarray(tokens[start:start + BATCH_SIZE].mean(axis=1, dtype=np.float32))
                           for start in range(0, len(tokens), BATCH_SIZE)], axis=0)


def batch_ranges(windows=60, batch_size=BATCH_SIZE):
    require(windows == 60 and batch_size == 8, 'Registered60 windows/batch8 cannot change')
    return [(start, min(start + batch_size, windows)) for start in range(0, windows, batch_size)]


def _save_arrays(path, arrays):
    require(not path.exists() and path.resolve().is_relative_to(FEATURES.resolve()), 'Refusing stale/outside feature output')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def run(device='cuda'):
    reg = registration()
    require(reg['budget']['features'] == FEATURE_BUDGET, 'Feature budget changed')
    config = reg['features']
    require(config['batch_size'] == BATCH_SIZE, 'Batch size changed')
    channels = config['channel_names']
    assets.canonical_channels(channels)
    source_manifest_path = checked(config['raw_manifest'])
    source_manifest = read(source_manifest_path)
    rows = source_manifest['records']
    ids = [row['subject_id'] for row in rows]
    cohort = read(WORK / 'S00/configs/cohort.json')
    require(ids == cohort['subject_ids'] and len(ids) == len(set(ids)) == 37
            and source_manifest['channel_names'] == channels, 'Registered37 native-order EEG inputs differ')
    projection_path = checked(config['projection'])
    matrix = np.load(projection_path, allow_pickle=False)
    expected_projection, projection_audit = assets.projection()
    require(matrix.dtype == np.float32 and np.array_equal(matrix, expected_projection), 'Saved projection is not the fixed label-free QR')
    # An already completed export is checked read-only; an incomplete directory
    # never automatically restarts any neural calls.
    if MANIFEST.exists():
        previous = read(MANIFEST)
        require(previous['passed'] and previous['registration'] == desc(OUT / 'registration.json'), 'Stale feature manifest')
        checked(previous['ledger'])
        for row in previous['records']:
            checked(row['features']); checked(row['sidecar'])
        return previous
    require(not FEATURES.exists() and not LEDGER.exists(), 'Partial feature export exists; no automatic neural rerun')
    require(device == reg['runtime']['device'] == 'cuda', 'Feature device differs from registered CUDA runtime')
    runtime(11)
    require(device != 'cuda' or torch.cuda.is_available(), 'Registered CUDA encoder runtime is unavailable')
    FEATURES.mkdir(parents=True, exist_ok=False)
    binding = desc(OUT / 'registration.json')
    ledger = {'registration': binding, 'status': 'started', 'started_at': now(),
        'budget': FEATURE_BUDGET, 'counts': {key: 0 for key in FEATURE_BUDGET}, 'attempts': [],
        'started_batches': {'eegpt': 0, 'labram': 0}, 'returned_batches': {'eegpt': 0, 'labram': 0},
        'formal_fits': 0, 'formal_updates': 0, 'encoder_gradient_passes': 0,
        'batch_size': 8, 'repeat_windows_per_person': 8, 'input_math': INPUT_MATH}
    save(LEDGER, ledger)
    from tridep_ccml.tridep_stage1.labram_assets import load_frozen_labram, channel_indices
    eegpt, eegpt_loading = assets.load_frozen_eegpt(device)
    labram, labram_loading = load_frozen_labram(device)
    models = {'eegpt': eegpt, 'labram': labram}
    before = {key: assets.encoder_state_digest(model) for key, model in models.items()}
    chan_ids = torch.as_tensor(assets.channel_ids(channels), dtype=torch.long, device=device)
    labram_chans = channel_indices(channels)
    ledger['load_audits'] = {'eegpt': eegpt_loading, 'labram': labram_loading}
    ledger['states_before'] = before
    save(LEDGER, ledger)

    def forward(encoder, phase, sid, start, end, values):
        count = end - start
        require(encoder in models and phase in ('first', 'repeat') and 1 <= count <= 8, 'Unknown forward accounting key')
        require(phase != 'repeat' or (start, end) == (0, 8), 'Repeat must exactly reproduce first batch8')
        attempt = {'encoder': encoder, 'phase': phase, 'subject_id': sid,
            'window_indices': list(range(start, end)), 'batch_size': count, 'started_at': now(), 'returned_at': None}
        ledger['attempts'].append(attempt); ledger['started_batches'][encoder] += 1
        save(LEDGER, ledger)
        tensor = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            if encoder == 'eegpt':
                output = eegpt(tensor, chan_ids=chan_ids)
                expected = (count, 16, 4, 512)
            else:
                output = labram.forward_features(tensor.reshape(count, 19, 4, 200), input_chans=labram_chans,
                    return_patch_tokens=True, return_all_tokens=False)
                expected = (count, 76, 200)
        # A returned neural call is counted even if its output validation later
        # fails; started/returned are invocation facts rather than quality gates.
        attempt['returned_at'] = now(); ledger['returned_batches'][encoder] += 1
        for key, increase in ((f'{encoder}_{phase}_windows', count), (f'{encoder}_total_windows', count), (f'{encoder}_batches', 1)):
            ledger['counts'][key] += increase
            require(ledger['counts'][key] <= FEATURE_BUDGET[key], 'Registered forward budget exceeded: ' + key)
        save(LEDGER, ledger)
        require(tuple(output.shape) == expected and output.dtype == torch.float32 and bool(torch.isfinite(output).all()),
                'Frozen encoder output contract failed: ' + encoder)
        result = np.ascontiguousarray(output.detach().cpu().numpy())
        return result

    records = []
    for row in rows:
        sid = row['subject_id']
        raw_path = checked(row['raw'])
        require(reg['sources'].get(row['raw']['path']) == row['raw']['sha256'], 'Raw EEG was not source-frozen')
        raw = np.load(raw_path, allow_pickle=False)
        adapted = paired_inputs(raw)
        initial, repeated = {}, {}
        for encoder in ('eegpt', 'labram'):
            blocks = [forward(encoder, 'first', sid, start, end, adapted[encoder][start:end]) for start, end in batch_ranges()]
            initial[encoder] = np.concatenate(blocks, axis=0)
            repeated[encoder] = forward(encoder, 'repeat', sid, 0, 8, adapted[encoder][:8])
            require(np.array_equal(initial[encoder][:8], repeated[encoder]), 'Same-batch8 repeated frozen encoder changed: ' + encoder)
        native, eegpt_features = eegpt_interface(initial['eegpt'], matrix)
        labram_features = labram_interface(initial['labram'])
        repeat_native, repeat_eegpt = eegpt_interface(repeated['eegpt'], matrix)
        repeat_labram = labram_interface(repeated['labram'])
        require(np.array_equal(eegpt_features[:8], repeat_eegpt) and np.array_equal(labram_features[:8], repeat_labram),
                'Repeated pooled/projected feature interface changed')
        arrays = {'eegpt': eegpt_features, 'labram': labram_features,
            'eegpt_input': adapted['eegpt'], 'labram_input': adapted['labram'],
            'eegpt_tokens': initial['eegpt'], 'labram_tokens': initial['labram'], 'eegpt_native512': native,
            'eegpt_repeat_tokens': repeated['eegpt'], 'labram_repeat_tokens': repeated['labram'],
            'eegpt_repeat_native512': repeat_native, 'eegpt_repeat': repeat_eegpt, 'labram_repeat': repeat_labram}
        require(eegpt_features.shape == labram_features.shape == (60, 200), 'Paired head feature shape changed')
        path = FEATURES / f'{sid}.npz'
        _save_arrays(path, arrays)
        sidecar = path.with_suffix('.json')
        shapes = {key: list(value.shape) for key, value in arrays.items()}
        save(sidecar, {'subject_id': sid, 'registration': binding, 'raw': row['raw'],
            'raw_manifest': config['raw_manifest'], 'features': desc(path), 'projection': config['projection'],
            'channel_names': channels, 'eegpt_canonical_channels': list(assets.CHANNEL_NAMES),
            'eegpt_channel_ids': list(assets.CHANNEL_IDS), 'input_math': INPUT_MATH, 'shapes': shapes,
            'dtype': 'float32', 'batch_size': 8, 'first_windows': 60, 'repeat_windows': 8,
            'repeat_exact': {'eegpt_tokens': True, 'labram_tokens': True, 'eegpt_features': True, 'labram_features': True},
            'frozen_eval': True, 'fit_subject_ids': [], 'projection_fitted': False,
            'producer': desc(Path(__file__))}, immutable=True)
        records.append({'subject_id': sid, 'features': desc(path), 'bundle': desc(path), 'sidecar': desc(sidecar), 'shapes': shapes})
        print('MATCHED4S FEATURES', sid, '60windows+8samebatchrepeat/encoder', flush=True)
    require(ledger['counts'] == FEATURE_BUDGET and ledger['started_batches'] == ledger['returned_batches'] == {'eegpt': 333, 'labram': 333},
            'Final paired encoder window/batch count differs')
    after = {key: assets.encoder_state_digest(model) for key, model in models.items()}
    require(after == before and all(not any(m.training for m in model.modules())
        and not any(p.requires_grad or p.grad is not None for p in model.parameters()) for model in models.values()),
        'Frozen encoder parameters/buffers or eval/gradient state changed')
    registration()  # Recheck every registered source after all frozen calls.
    ledger.update(status='complete', finished_at=now(), states_after=after)
    save(LEDGER, ledger)
    result = {'passed': True, 'registration': binding, 'raw_manifest': config['raw_manifest'],
        'projection': config['projection'], 'projection_audit': projection_audit, 'ledger': desc(LEDGER),
        'records': records, 'subject_ids': ids, 'channel_names': channels, 'input_math': INPUT_MATH,
        'frozen_states_before': before, 'frozen_states_after': after, 'frozen_states_exact': True,
        'counts': ledger['counts'], 'batch_size': 8, 'repeat_windows_per_person': 8,
        'formal_fits': 0, 'formal_updates': 0, 'producer': desc(Path(__file__))}
    save(MANIFEST, result, immutable=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    run(parser.parse_args().device)
