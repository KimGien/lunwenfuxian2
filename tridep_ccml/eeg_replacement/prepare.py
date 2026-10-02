"""Measure full batches, lock all candidates before any formal fit or score."""
from __future__ import annotations
import argparse
import io
import platform
import shutil
import sys
import time
import unittest
from .common import *
from .data import FoldData, check_splits
from .models import R1Model, structure_manifest
from .train import step, orders


def measure():
    from .test_contracts import Contracts
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    print(log.getvalue(), flush=True)
    require(result.wasSuccessful(), 'R1 contracts failed')
    tested = ['common.py','models.py','data.py','train.py','test_contracts.py','prepare.py']
    write_new(OUT/'test_results.json', {'passed':True, 'tests':result.testsRun, 'failures':0, 'errors':0,
        'sources':{name:descriptor(CODE/name) for name in tested}, 'log':log.getvalue()})
    context = RecordContext()
    check_splits(context)
    write_new(OUT/'structure_manifest.json', structure_manifest())
    data = FoldData(1)
    measurements = {}
    for condition in CONDITIONS:
        set_seed(11)
        model = R1Model(condition,11).cuda().train()
        optimizer = model.optimizer()
        ids = data.splits['train']
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        batch_seconds = []
        for offset in range(0,len(ids),8):
            inputs, labels = data.batch(ids[offset:offset+8],condition)
            t = time.perf_counter()
            step(model,optimizer,inputs,labels,1)
            torch.cuda.synchronize()
            batch_seconds.append(time.perf_counter()-t)
        elapsed = time.perf_counter()-start
        gradients = {name:float(p.grad.detach().norm()) for name,p in model.named_parameters()}
        model.eval()
        before = state_digest(model.state_dict())
        sample, _ = data.batch(ids[:8],condition)
        with torch.inference_mode():
            output = model(sample)
            reps = model.representations(sample)
            singles = torch.cat([model({k:v[i:i+1] for k,v in sample.items()}) for i in range(8)])
        require(output.shape == ((8,3,2) if model.joint else (8,2)), 'Output shape mismatch')
        require(reps.shape == ((8,3,128) if model.joint else (8,128)), '128D interface mismatch')
        torch.testing.assert_close(output,singles,rtol=1e-4,atol=1e-5)
        require(state_digest(model.state_dict()) == before, 'Inference updates model')
        measurements[condition] = {'full_train_epoch_seconds':elapsed, 'batch_seconds':batch_seconds,
            'estimate_15_runs_seconds':elapsed*60*15, 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'parameters':model.counts(),
            'all_parameter_gradients_finite':True, 'gradient_norms':gradients,
            'inference_batch_max_abs':float((output-singles).abs().max()), 'inference_state_unchanged':True,
            'output_shape':list(output.shape), 'representation_shape':list(reps.shape)}
        print(f'MEASURE {condition} epoch={elapsed:.3f}s peak={torch.cuda.max_memory_allocated()/2**20:.1f}MiB',flush=True)
        del model, optimizer, inputs, output, reps, singles, sample
        torch.cuda.empty_cache()
    require(measurements['U1']['parameters']['eeg_including_head'] <= 50000 and
            measurements['C1']['parameters']['eeg_including_head'] <= 50000, 'EEGNet budget exceeded')
    cache_bytes = sum(p.stat().st_size for p in (OUT/'cache').rglob('*') if p.is_file())
    estimates = [sum(measurements[c]['estimate_15_runs_seconds'] for c in stage) for stage in STAGES]
    info = {'passed':True, 'created_utc':now(), 'augmentation':False, 'measurements':measurements,
        'stage_estimated_seconds':estimates, 'total_training_estimated_seconds':sum(estimates),
        'estimate_scope':'one real fold of 27 train subjects, one epoch per condition; initialization/warmup included; excludes preparation/evaluation/verification',
        'smoke_updates':32, 'smoke_models_saved_for_formal_training':False,
        'contract_tests':descriptor(OUT/'test_results.json'), 'cache_bytes':cache_bytes,
        'estimated_checkpoint_bytes':sum(m['parameters']['total']*4*3*15 for m in measurements.values()),
        'disk_free_bytes':shutil.disk_usage(OUT).free,
        'runtime':{'python':platform.python_version(), 'executable':sys.executable,
            'torch':torch.__version__, 'numpy':np.__version__, 'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(), 'device':'cuda:0', 'dtype':'float32',
            'deterministic':True, 'matmul_tf32':False, 'cudnn_tf32':False,
            'mixed_precision':False, 'subject_microbatching':False, 'threads':torch.get_num_threads()}}
    write_new(OUT/'preflight.json',info)
    print(f'PRELIGHT passed; estimated training hours={sum(estimates)/3600:.3f}',flush=True)


def lock():
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'], 'Missing passing tests')
    for d in tests['sources'].values():
        verify_descriptor(d)
    preflight = read_json(OUT/'preflight.json')
    require(preflight['passed'], 'Preflight failed')
    context = RecordContext()
    history = {}
    for folder in [TC/'v2', TC/'eeg_aug', TC/'reports', TC/'configs', TC/'manifests']:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                history[p.relative_to(WORK).as_posix()] = digest(p)
    for p in TC.glob('*.py'):
        history[p.relative_to(WORK).as_posix()] = digest(p)
    for parent in (TC/'experiments').iterdir():
        if parent.is_dir() and parent != OUT:
            for p in parent.glob('*.json'):
                history[p.relative_to(WORK).as_posix()] = digest(p)
    write_new(OUT/'history_before.json',history)
    cfg = {'experiment':'EEG-R1-37-fulltext', 'version':1, 'conditions':CONDITIONS,
        'stages':STAGES, 'seeds':SEEDS, 'folds':FOLDS, 'subject_ids':context.ids,
        'cohort_sha256':context.cohort_sha256, 'folds_sha256':context.folds_sha256,
        'augmentation':False, 'epochs':60, 'batch_size_subjects':8, 'windows_per_subject':60,
        'drop_last':False, 'optimizer':'Adamax', 'U_lr':1e-3, 'U_weight_decay_all_parameters':1e-4,
        'C_encoder_projection_lr':1e-4, 'C_head_lr':1e-3, 'C_weight_decay':0.,
        'gradient_clip_global_norm':1., 'ccml':LOSS, 'threshold':.5, 'tie_class':0,
        'checkpoint_selection':'fixed_epoch_60', 'updates_per_run':240, 'total_runs':120, 'total_updates':28800,
        'early_stopping':False, 'class_resampling':False, 'outer_selection':False,
        'outer_score_release':'after all 120 fits and condition-wide checkpoint verification; no outer inference in trainer',
        'data_manifest':descriptor(OUT/'data_manifest.json'), 'structure_manifest':descriptor(OUT/'structure_manifest.json'),
        'preflight':descriptor(OUT/'preflight.json'), 'sources':source_hashes(),
        'checkpoint_reload_tolerance':{'rtol':1e-5,'atol':1e-6},
        'metric_recompute_atol':1e-12, 'runtime':preflight['runtime'],
        'positive_label':'MDD', 'statistics':'mean folds within seed, then mean of 3 seeds; pooled OOF separate',
        'usability_collapse_policy':'conservative: any single-class test fold fails usability; disclose fold and pooled collapse counts',
        'shared_weights':'S-control projection RNG order and seed-specific evidence heads; explicit tensor equality across C conditions',
        'randomness':'independent CPU generator for subject orders; model dropout uses reset seed per run',
        'input_processing':'native original EEG after registered fixed preprocessing; train-only channel standardization for U1/C1/U2/C2',
        'frozen_backbones':'CBraMod/HuBERT/MacBERT label-free cached outputs; no backbone in optimizer',
        'plan':descriptor(PLAN), 'locked_utc':now()}
    if (OUT/'configuration.json').exists():
        return verify_lock()
    write_new(OUT/'configuration.json',cfg)
    print('LOCKED all 8 conditions; 120 runs; no EEG augmentation; no outer scores inspected.',flush=True)


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock',action='store_true')
    args = parser.parse_args()
    lock() if args.lock else measure()
