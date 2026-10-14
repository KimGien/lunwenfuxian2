"""Analyze one existing py-spy JSON; never attach, query a process or import tensor code."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
PROFILE = SOURCE.with_name('readonly_cpu_stack_csbrain_20261009_2239.json')
PROFILE_SHA = '1ee74bf7d47bd58186475b176a8643092e3cdbe3300873f0e598158129270bb2'


def descriptor(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def normalize(frame):
    return frame.get('file', '').replace('\\', '/')


def analyze(observer=None):
    binding = descriptor(PROFILE)
    assert binding['sha256'] == PROFILE_SHA
    value = json.loads(PROFILE.read_text(encoding='utf-8'))
    frames = value['shared']['frames']
    profiles = value['profiles']
    assert value['exporter'] == 'py-spy@0.4.2'
    assert all(p['type'] == 'sampled' and p['unit'] == 'seconds' for p in profiles)
    assert all(len(p['samples']) == len(p['weights']) for p in profiles)
    assert all(type(i) is int and 0 <= i < len(frames) for p in profiles for s in p['samples'] for i in s)
    assert all(not isinstance(w, bool) and math.isfinite(w) and w > 0 for p in profiles for w in p['weights'])
    main = next(p for p in profiles if p['name'] == 'Thread 48552 "MainThread"')
    samples, weights = main['samples'], main['weights']
    assert len(samples) == 1524 and all(w == .02 for w in weights)
    groups = {
        'state_digest_common': lambda f: normalize(f).endswith('/tridep_stage1/common.py') and f['name'] == 'state_digest',
        'state_digest_D2H_line91': lambda f: normalize(f).endswith('/tridep_stage1/common.py') and f['line'] == 91,
        'state_digest_numpy_bytes_line95': lambda f: normalize(f).endswith('/tridep_stage1/common.py') and f['line'] == 95,
        'checkpoint_cpu_state': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['name'] == 'cpu_state',
        'region_mask_to_line108': lambda f: normalize(f).endswith('/CSBrain_transformerlayer.py') and f['line'] == 108,
        'F1_adapt_line114': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['line'] == 114,
        'F1_as_tensor_H2D_line117': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['line'] == 117,
        'F1_representation_finite_line120': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['line'] == 120,
        'F1_diagnostics': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['name'] == 'diagnostics',
        'F1_subject_update': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['name'] == 'subject_update',
        'F1_fit': lambda f: normalize(f).endswith('/eeg_all_candidates/finetune.py') and f['name'] == 'fit',
        'torch_padding_call_boundary': lambda f: '/torch/' in normalize(f) and f['name'] == 'pad',
        'torch_convolution_call_boundary': lambda f: f['name'] == '_conv_forward',
        'torch_attention_call_boundary': lambda f: f['name'] == 'multi_head_attention_forward',
        'autograd_engine_call_boundary': lambda f: f['name'] == '_engine_run_backward',
        'CSBrain_region_attention': lambda f: f['name'] == '_inter_region_attention_static',
    }
    rows = []
    for name, predicate in groups.items():
        matched = {i for i, frame in enumerate(frames) if predicate(frame)}
        included = [i for i, stack in enumerate(samples) if matched.intersection(stack)]
        leaves = [i for i, stack in enumerate(samples) if stack and stack[-1] in matched]
        rows.append({'group': name, 'frame_definitions': [frames[i] for i in sorted(matched)],
                     'inclusive_recorded_samples': len(included), 'leaf_recorded_samples': len(leaves),
                     'inclusive_fraction_of_1524_recorded_main_stacks': len(included) / len(samples),
                     'leaf_fraction_of_1524_recorded_main_stacks': len(leaves) / len(samples),
                     'inclusive_weighted_sample_axis_seconds_proxy': math.fsum(weights[i] for i in included),
                     'leaf_weighted_sample_axis_seconds_proxy': math.fsum(weights[i] for i in leaves),
                     'nominal_2000_slot_missing_assignment_fraction_bounds': [len(included) / 2000, (len(included) + 476) / 2000],
                     'nominal_slot_upper_sensitivity_also_allowing_32_shallow_or_other_stacks': min(1.0, (len(included) + 476 + 32) / 2000)})
    leaf_counts = Counter(stack[-1] for stack in samples if stack)
    top_leaf = [{'frame': frames[i], 'recorded_samples': n, 'recorded_main_stack_fraction': n / len(samples)}
                for i, n in leaf_counts.most_common(30)]
    quality = []
    for p in profiles:
        attributed = sum(any(frames[i].get('file') for i in stack) for stack in p['samples'])
        quality.append({'thread': p['name'], 'recorded_samples': len(p['samples']),
                        'samples_with_file_attributed_Python_frame': attributed,
                        'synthetic_thread_label_only_samples': sum(all(not frames[i].get('file') for i in stack) for stack in p['samples']),
                        'weighted_sample_axis_seconds': math.fsum(p['weights']),
                        'not_thread_cpu_seconds': True})
    source_paths = ['tridep_ccml/tridep_stage1/common.py', 'tridep_ccml/eeg_all_candidates/finetune.py',
                    'tridep_ccml/eeg_all_candidates/finetune_v3.py',
                    'tridep_ccml/eeg_all_candidates/vendor/csbrain/models/CSBrain_transformerlayer.py',
                    'tridep_ccml/eeg_all_candidates/vendor/csbrain/models/CSBrain_transformer.py',
                    'S00/.venv/Lib/site-packages/torch/nn/functional.py',
                    'S00/.venv/Lib/site-packages/torch/nn/modules/conv.py']
    result = {'schema': 'readonly_cpu_stack_analysis_v1', 'analysis_completed': True,
              'source': descriptor(SOURCE), 'profile': binding,
              'observer_run_evidence': descriptor(observer) if observer else None,
              'observer_run_evidence_scope': 'Separately written controller summary; profile does not contain failed sample counters.',
              'sampled_source_files': [descriptor(WORK / p) for p in source_paths],
              'method': 'Each sample is root-to-leaf. Inclusive presence is counted once per group per stack, including recursive/repeated frames only once; leaf is the final frame. Nested groups overlap and must not be added. Weights are exporter sampling-axis proxies only.',
              'acquisition_reported_by_root': {'requested_duration_seconds': 40, 'rate_hz': 50,
                  'flags': ['--idle', '--threads', '--nonblocking', '--full-filenames'],
                  'pid': 32880, 'profiler_exit_code': 0, 'stdout_total_thread_samples': 3048,
                  'stdout_read_errors': 475, 'warning_thread_id': 48552,
                  'full_stderr_preserved': False, 'actual_wall_timer_certificate': None},
              'quality': {'frames': len(frames), 'profiles': quality,
                  'main_recorded_samples': 1524, 'nominal_requested_sampling_slots': 2000,
                  'recorded_to_nominal_slot_ratio': 1524 / 2000,
                  'reported_successful_rounds_plus_read_errors': 1524 + 475,
                  'recorded_to_success_plus_reported_errors_ratio': 1524 / (1524 + 475),
                  'nominal_unrepresented_slots': 476,
                  'main_stacks_without_F1_fit_ancestry': 1524 - next(row['inclusive_recorded_samples'] for row in rows if row['group'] == 'F1_fit'),
                  'missingness_bias_unknown': True, 'nonblocking_stack_internal_consistency_not_guaranteed': True},
              'attribution': rows, 'main_top_leaf_frames': top_leaf,
              'interpretation': [
                  'The window mostly samples diagnostics. It is not a complete epoch or representative full-plan phase mix.',
                  'A Python call boundary can encompass native CPU work, kernel launch, synchronization or waiting. No native stack or CUDA timeline was captured, so CuDNN/kernel busy time and CPU-vs-GPU causality cannot be separated.',
                  'The profile sees state_digest, mask.to and transfer sites, but their recorded presence is small relative to padding, convolution and attention paths. This does not establish any one site as the dominant bottleneck.',
                  'F1 line117 is host-to-device as_tensor; line120 checks finite representations. These must not be conflated.',
                  'Read errors may correlate with execution state. Slot bounds only assign nominal unrepresented slots adversarially and are not confidence intervals, wall-time estimates or CPU/GPU utilization bounds.',
                  'Thirty-two recorded main stacks lack F1.fit ancestry; this may reflect partial nonblocking unwind or other interpreter activity. They are not silently discarded or automatically called invalid.',
                  'The second thread has only synthetic labels and supplies no Python function attribution. Thread sampling axes must not be summed into process wall time.',
                  'No static operation count, measured speedup, per-epoch time or optimization correctness is inferred from sampling frequency.'
              ],
              'actions_by_this_analysis': {'profiler_launches': 0, 'process_queries_or_controls': 0,
                  'GPU_queries_or_calls': 0, 'model_constructions': 0, 'raw_feature_checkpoint_or_prediction_reads': 0},
              'full_plan_complete_claimed': False}
    assert descriptor(PROFILE) == binding
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observer', type=Path)
    args = parser.parse_args()
    result = analyze(args.observer)
    target = SOURCE.with_name('readonly_cpu_stack_analysis_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'analysis': descriptor(target), 'main_recorded_samples': result['quality']['main_recorded_samples'],
                      'attribution': [{k: v for k, v in row.items() if k != 'frame_definitions'} for row in result['attribution']]}))
