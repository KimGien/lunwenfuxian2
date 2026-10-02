"""Recover only previously empty short-audio questions with the locked HuBERT."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
from pathlib import Path

from . import audio as parent
from .common import (WORK, TC, RecordContext, descriptor, verify_descriptor, read_json,
                     require, write_new, set_seed, digest)
from tridep_ccml.train import state_digest
import numpy as np
import torch

OUT = TC/'experiments/CCML-v2-AS-37-fulltext'
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/CCML_v2_AS_短音频覆盖修订固定计划.md'
STAGES = {s: f'CCML-v2-AS-{s}-37-fulltext' for s in ['adapt', 'control', 'N']}
VENDOR = WORK/'S00/upstream-dev/lib'


def configuration(context):
    cfg = parent.configuration(context)
    cfg.update(family=OUT.name, plan=descriptor(PLAN), stages=STAGES,
               paired_A_configuration=descriptor(parent.CONFIG))
    cfg['speech'] = {**cfg['speech'], 'short_question': 'one actual-length HuBERT vector per previously empty question; denominator remains 29'}
    cfg['comparisons'] = {'primary': 'AS-N versus A-N', 'paired': ['AS-adapt versus A-adapt', 'AS-control versus A-control'],
                          'within_version': 'AS-N versus AS-control', 'old_N_NP': 'development references; no automatic replacement'}
    cfg['prior_status']['A'] = descriptor(parent.OUT/'status.json')
    cfg['short_extraction'] = {'device': 'cuda', 'dtype': 'float32', 'batch_size': 1,
        'sample_rate': 16000, 'minimum_samples': 400, 'maximum_samples_exclusive': 80000,
        'padding': 'none', 'window': 'complete preprocessed waveform once',
        'pcm': 'PCM16 memory roundtrip, same as historical extraction',
        'frozen_backbone': True, 'dropout': 'eval', 'cudnn_tf32': True, 'matmul_tf32': False,
        'deterministic': True, 'long_controls': 'first nonempty question, first window, each of 37 subjects',
        'control_rtol': 2e-5, 'control_atol': 2e-5, 'short_repeat': 'bitwise equal',
        'expected_repaired_questions': 96, 'expected_unchanged_questions': 977,
        'expected_affected_subjects': 29, 'expected_unchanged_subjects': 8}
    paths = [Path(__file__), TC/'v2/short_train.py', TC/'v2/complete_short.py', TC/'v2/test_short.py',
             TC/'v2/audio.py', TC/'v2/complete_audio.py', TC/'v2/common.py', TC/'v2/nr.py',
             TC/'v2/rebuild_eeg.py', TC/'ccml.py', TC/'train.py', TC/'records.py', TC/'metrics.py',
             VENDOR/'preprocessing/speech_preprocessing.py', VENDOR/'feature_extraction/speech/chinese_hubert.py']
    cfg['sources'] = {p.relative_to(WORK).as_posix(): descriptor(p) for p in paths}
    return cfg


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        tracked = read_json(parent.OUT/'history_before.json')
        for p in parent.OUT.rglob('*'):
            if p.is_file():
                tracked[p.relative_to(WORK).as_posix()] = digest(p)
        for p in [TC/'v2/audio.py', TC/'v2/complete_audio.py', TC/'v2/test_audio.py', parent.PLAN,
                  TC/'experiments/CCML-v2-A-training.log', TC/'experiments/CCML-v2-A-preflight.log']:
            tracked[p.relative_to(WORK).as_posix()] = digest(p)
        write_new(path, tracked)
    tracked = read_json(path)
    for name, value in tracked.items():
        require(digest(WORK/name) == value, f'Historical artifact changed: {name}')
    return len(tracked)


def setup():
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    protect_history()
    cfg = configuration(context)
    for item in cfg['prior_status'].values():
        status = read_json(verify_descriptor(item))
        require(status['status'] == 'complete', 'Preceding stage incomplete')
        verify_descriptor(status['audit'])
        verify_descriptor(status['report'])
    write_new(CONFIG, cfg)
    return context, cfg


def vendor_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, VENDOR/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess(raw, expected_samples):
    import librosa
    module = vendor_module('preprocessing/speech_preprocessing.py', 'as_vendor_preprocess')
    y, _ = librosa.load(raw, sr=16000, mono=True)
    require(y.ndim == 1 and y.size > 0 and np.isfinite(y).all(), 'Invalid raw waveform')
    y = module.preprocess_audio(y)
    require(y.dtype == np.float32 and len(y) == expected_samples and np.isfinite(y).all(), 'Preprocessed length or values changed')
    return y


def short_window(y):
    require(y.ndim == 1 and y.dtype == np.float32 and np.isfinite(y).all(), 'Invalid short waveform')
    require(400 <= len(y) < 80000, 'Short waveform outside registered bounds')
    return y  # Real length; no padding, duplication, or stretching.


def pcm16_roundtrip(y):
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, y, 16000, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    output, sr = sf.read(buffer, dtype='float32')
    require(sr == 16000 and output.shape == y.shape and np.isfinite(output).all(), 'PCM roundtrip mismatch')
    return output


def waveform_hash(y):
    return hashlib.sha256(np.ascontiguousarray(y, dtype='<f4').tobytes()).hexdigest()


def load_extractor():
    set_seed(11)
    require(torch.cuda.is_available(), 'Registered CUDA extraction device unavailable')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone = read_json(WORK/'S00/manifests/backbones.json')['hubert_code']
    for filename, info in backbone['files'].items():
        require(digest(WORK/'S00/models/hubert_code'/filename) == info['sha256'], 'HuBERT file changed')
    cls = vendor_module('feature_extraction/speech/chinese_hubert.py', 'as_vendor_hubert').HubertChineseExtractor
    extractor = cls(str(WORK/'S00/models/hubert_code'), torch.device('cuda'))
    extractor.model.requires_grad_(False).eval()
    saved = torch.load(WORK/'S00/models/hubert_code/pytorch_model.bin', map_location='cpu', weights_only=True)
    mappings = {}
    for name, value in extractor.model.state_dict().items():
        original = name
        if original not in saved:
            original = original.replace('.parametrizations.weight.original0', '.weight_g').replace('.parametrizations.weight.original1', '.weight_v')
        require(original in saved and torch.equal(value.detach().cpu(), saved[original]), f'Unmatched retained HuBERT weight: {name}')
        mappings[name] = original
    del saved
    state_hash = state_digest(extractor.model.state_dict())
    runtime = {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
               'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32),
               'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
               'state_sha256': state_hash, 'retained_weight_mapping': mappings,
               'all_parameters_frozen': all(not p.requires_grad for p in extractor.model.parameters()),
               'all_modules_eval': all(not m.training for m in extractor.model.modules())}
    return extractor, runtime


def embed(extractor, waveform, repeat=False):
    values = extractor.processor(waveform, sampling_rate=16000, return_tensors='pt',
                                 padding='longest', truncation=True, max_length=80000).input_values
    require(values.shape == (1, len(waveform)), 'Processor padded or truncated actual-length input')
    input_hash = waveform_hash(values.numpy())
    values = values.to('cuda')
    with torch.no_grad():
        hidden = extractor.model(values)
        if repeat:
            require(torch.equal(hidden, extractor.model(values)), 'Repeated frozen inference differs')
    n = len(waveform)
    for kernel, stride in zip(extractor.config.conv_kernel, extractor.config.conv_stride):
        n = (n-kernel)//stride+1
    require(hidden.shape == (1, n, 768) and n > 0 and torch.isfinite(hidden).all(), 'Invalid valid-frame shape')
    vector = hidden.squeeze(0).mean(0).cpu().numpy()
    require(vector.dtype == np.float32 and np.isfinite(vector).all(), 'Invalid embedding')
    return vector, {'pcm_samples': len(waveform), 'valid_frames': n, 'pcm_sha256': waveform_hash(waveform),
                    'processor_input_sha256': input_hash}


def replace_short_means(old_questions, old_counts, replacement):
    require(old_questions.shape == (37, 29, 768) and old_counts.shape == (37, 29), 'Unexpected pool shapes')
    empty = old_counts == 0
    require(set(replacement) == set(zip(*np.where(empty))), 'Replacement set differs from empty questions')
    result, counts = old_questions.copy(), old_counts.copy()
    for (i, q), value in replacement.items():
        require(value.shape == (768,) and value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid replacement')
        result[i, q] = value
        counts[i, q] = 1
    require(np.array_equal(result[~empty], old_questions[~empty]), 'Long-question mean changed')
    return result, counts


def extract_pool(context, cfg):
    previous = read_json(parent.OUT/'speech_pool.json')
    with np.load(verify_descriptor(previous['cache']), allow_pickle=False) as archive:
        old = {k: archive[k].copy() for k in archive.files}
    require(old['subject_ids'].tolist() == context.ids, 'Parent pool IDs changed')
    extractor, runtime = load_extractor()
    long_checks, repaired, replacements = [], [], {}
    lookup = {s: i for i, s in enumerate(context.ids)}
    for sid in context.ids:
        row = next(r for r in previous['sources'] if r['subject_id'] == sid and not r['empty'])
        metadata = read_json(verify_descriptor(row['metadata']))
        y = preprocess(verify_descriptor(row['raw_audio']), row['processed_samples'])
        start, end = metadata['windows_samples'][0]
        waveform = pcm16_roundtrip(y[start:end])
        vector, inputs = embed(extractor, waveform)
        reference = np.load(verify_descriptor(row['cache']), allow_pickle=False)[0]
        error = float(np.max(np.abs(vector-reference)))
        require(np.allclose(vector, reference, rtol=cfg['short_extraction']['control_rtol'],
                            atol=cfg['short_extraction']['control_atol']), f'Long extraction control mismatch {sid}: max_abs={error}')
        long_checks.append({'subject_id': sid, 'question': row['question'], 'cache': row['cache'],
                            'window': [start, end], 'max_abs_error': error, **inputs})
        if len(long_checks) % 10 == 0:
            print(f'Long extraction controls verified: {len(long_checks)}/37', flush=True)
    for row in previous['sources']:
        if not row['empty']:
            continue
        verify_descriptor(row['metadata'])
        old_cache = np.load(verify_descriptor(row['cache']), allow_pickle=False)
        require(old_cache.shape == (0, 768), 'Replacement source was not empty')
        sid, q = row['subject_id'], row['question']
        y = short_window(preprocess(verify_descriptor(row['raw_audio']), row['processed_samples']))
        waveform = pcm16_roundtrip(y)
        vector, inputs = embed(extractor, waveform, repeat=True)
        path = OUT/'short_features'/sid/f'{q:02d}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(np.array_equal(np.load(path, allow_pickle=False), vector), 'Existing short vector changed')
        else:
            with path.open('xb') as stream:
                np.save(stream, vector, allow_pickle=False)
        meta = {'subject_id': sid, 'question': q, 'configuration': descriptor(CONFIG), 'feature': descriptor(path),
                'previous_source': row, 'fit_subject_ids': [], 'labels_used': False, 'padding_samples': 0,
                'window_samples': [0, len(y)], 'backbone_state_sha256': runtime['state_sha256'],
                'exact_repeated_inference': True, **inputs}
        write_new(path.with_suffix('.json'), meta)
        replacements[lookup[sid], q-1] = vector
        repaired.append({'subject_id': sid, 'question': q, 'feature': descriptor(path),
                         'metadata': descriptor(path.with_suffix('.json'))})
        if len(repaired) % 12 == 0:
            print(f'Short questions extracted: {len(repaired)}/96', flush=True)
    require(len(repaired) == 96 and len(long_checks) == 37, 'Incomplete extraction')
    require(state_digest(extractor.model.state_dict()) == runtime['state_sha256'], 'Frozen HuBERT weights changed')
    questions, counts = replace_short_means(old['question_means'], old['segment_counts'], replacements)
    pooled = np.stack([parent.pool_subject(q) for q in questions])
    untouched = ~(old['segment_counts'] == 0).any(axis=1)
    require(int(untouched.sum()) == 8 and np.array_equal(pooled[untouched], old['pooled'][untouched]), 'Unaffected pooled subjects changed')
    require((counts > 0).all(), 'Missing questions remain')
    parent.save_npz(OUT/'speech_pool.npz', {'question_means': questions, 'segment_counts': counts,
                    'pooled': pooled, 'subject_ids': np.array(context.ids)})
    write_new(OUT/'speech_pool.json', {'cache': descriptor(OUT/'speech_pool.npz'), 'configuration': descriptor(CONFIG),
              'cohort_sha256': context.cohort_sha256, 'subject_ids': context.ids, 'fit_subject_ids': [], 'labels_used': False,
              'parent_manifest': descriptor(parent.OUT/'speech_pool.json'), 'repaired': repaired,
              'short_audio_repaired': True, 'questions': 1073, 'repaired_questions': 96, 'unchanged_questions': 977,
              'affected_subject_ids': [s for i, s in enumerate(context.ids) if not untouched[i]],
              'unchanged_subject_ids': [s for i, s in enumerate(context.ids) if untouched[i]], 'empty_questions': 0})
    write_new(OUT/'extraction_audit.json', {'passed': True, 'configuration': descriptor(CONFIG), 'runtime': runtime,
              'long_controls': long_checks, 'long_control_max_abs_error': max(r['max_abs_error'] for r in long_checks),
              'repaired_questions': 96, 'all_short_repeat_exact': True, 'frozen_backbone_unchanged': True,
              'long_question_means_unchanged': 977, 'unaffected_pooled_subjects_unchanged': 8})
    print('Coverage complete: 1073/1073 questions, 96 repaired, 977 unchanged.', flush=True)
    del extractor
    torch.cuda.empty_cache()


def load_pool(context):
    info = read_json(OUT/'speech_pool.json')
    require(info['configuration'] == descriptor(CONFIG) and info['cohort_sha256'] == context.cohort_sha256 and
            info['subject_ids'] == context.ids and info['fit_subject_ids'] == [] and info['labels_used'] is False,
            'AS pool provenance mismatch')
    with np.load(verify_descriptor(info['cache']), allow_pickle=False) as arrays:
        pooled = arrays['pooled'].copy()
        require(arrays['subject_ids'].tolist() == context.ids and (arrays['segment_counts'] > 0).all(), 'AS coverage/identity mismatch')
    require(pooled.shape == (37, 768) and pooled.dtype == np.float32 and np.isfinite(pooled).all(), 'Invalid AS pool')
    return pooled, {'cache': info['cache'], 'manifest': descriptor(OUT/'speech_pool.json')}


def verify_short_reextraction():
    info = read_json(OUT/'speech_pool.json')
    expected_runtime = read_json(OUT/'extraction_audit.json')['runtime']
    extractor, runtime = load_extractor()
    require(runtime == expected_runtime, 'Extraction runtime/state changed')
    for i, row in enumerate(info['repaired']):
        meta = read_json(verify_descriptor(row['metadata']))
        require(meta['configuration'] == descriptor(CONFIG), 'Short feature config changed')
        source = meta['previous_source']
        waveform = pcm16_roundtrip(short_window(preprocess(verify_descriptor(source['raw_audio']), source['processed_samples'])))
        vector, inputs = embed(extractor, waveform)
        require(all(meta[k] == v for k, v in inputs.items()), 'Short extraction inputs changed')
        require(np.array_equal(vector, np.load(verify_descriptor(row['feature']), allow_pickle=False)), 'Short embedding does not exactly reproduce')
        if (i+1) % 24 == 0:
            print(f'Short raw-audio reextraction verified: {i+1}/96', flush=True)
    require(state_digest(extractor.model.state_dict()) == runtime['state_sha256'], 'Frozen backbone changed during replay')
    del extractor
    torch.cuda.empty_cache()
    return {'raw_audio_reextractions_exact': len(info['repaired']), 'same_frozen_backbone': True,
            'extraction_source': descriptor(OUT/'extraction_audit.json')}


def preflight():
    context, cfg = setup()
    extract_pool(context, cfg)
    from . import short_train
    checks = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            for stage in STAGES:
                set_seed(seed)
                data = parent.load_data(context, seed, fold, stage)
                model = parent.make_model(seed, stage)
                initial_hash = state_digest(model.state_dict())
                folder = parent.OUT/'runs'/parent.STAGES[stage]/f'seed_{seed}/fold_{fold}'
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                saved = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
                require(saved['metadata']['initial_state_sha256'] == initial_hash ==
                        state_digest(short_train.make_model(seed, stage).state_dict()), 'A/AS initialization differs')
                model.load_state_dict(saved['state_dict'], strict=True)
                require(parent.predictions(model, data, context, seed, fold, stage, folder/'checkpoint.pt') ==
                        read_json(folder/'predictions.json'), 'A predictions do not exactly reproduce')
                checks.append({'seed': seed, 'fold': fold, 'stage': stage, 'exact_reproduced_rows': 37,
                               'paired_initialization_sha256': initial_hash})
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG),
              'checks': checks, 'exact_A_records': 1665, 'pool': descriptor(OUT/'speech_pool.json'),
              'extraction': descriptor(OUT/'extraction_audit.json'), 'protected_files_unchanged': protect_history()})
    print('AS preflight passed: extraction coverage and 1665 exact A-stage records.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'verify-extraction'])
    (preflight if parser.parse_args().action == 'preflight' else verify_short_reextraction)()
