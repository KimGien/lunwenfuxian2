"""One-GPU queue for currently implemented frozen candidates and controls.

The registration still retains all360 formal fits, including outstanding raw
supervised/SSL and fine-tuning routes. Completing this batch does not release
external tests or mark any partially covered paper complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from . import registry as R, storage as S

METHODS = ('EEGPT', 'CSBrain', 'REVE', 'LUNA', 'CodeBrain', 'BIOT')
CONTROLS = ('CBraMod-frozen-smallhead', 'LaBraM-native5s-frozen-smallhead', 'LaBraM-native4s-frozen-smallhead')
DIAGNOSTICS = ('EEGPT-mean-projection4s-originalCNN-300ep', 'LaBraM-native4s-originalCNN-300ep')


def run(*, wait_paired=False, resume=False):
    R.load_registration()
    registration_path = S.OUT / 'ready_queue/registration.json'
    if not registration_path.exists():
        sources = [S.desc(S.WORK / ('tridep_ccml/eeg_all_candidates/' + name)) for name in (
            'ready_queue.py', 'frozen_heads.py', 'export_features.py', 'storage.py')]
        sources.extend(S.desc(S.WORK / ('tridep_ccml/eeg_all_candidates/adapters/' + name)) for name in (
            'csbrain.py', 'reve.py', 'luna.py', 'codebrain.py', 'biot.py', '_author_support.py'))
        S.save(registration_path, {'registry': R.registration_descriptor(), 'sources': sources,
               'frozen_candidate_methods': list(METHODS), 'control_conditions': list(CONTROLS),
               'formal_fits_in_this_batch': 135, 'test_forward_allowed': False,
               'registered_at': S.now(), 'wait_for_all30_paired_fits': True}, immutable=True)
    for source in S.read(registration_path)['sources']:
        S.checked(source)
    if wait_paired:
        S.save(S.OUT / 'ready_queue/status.json', {'status': 'waiting_for_paired30', 'updated_at': S.now(),
               'formal_fits_in_this_batch': 135, 'test_evaluation_frozen': True})
        while any(R.fit_status(condition, seed, fold)['status'] != 'training_complete'
                  for condition in DIAGNOSTICS for seed in (11, 22, 33) for fold in range(1, 6)):
            if any(R.fit_status(condition, seed, fold)['status'] == 'failed'
                   for condition in DIAGNOSTICS for seed in (11, 22, 33) for fold in range(1, 6)):
                raise ValueError('Paired training failed; preserve evidence and repair before starting the next GPU batch')
            time.sleep(10)
    else:
        if any(R.fit_status(condition, seed, fold)['status'] != 'training_complete'
               for condition in DIAGNOSTICS for seed in (11, 22, 33) for fold in range(1, 6)):
            raise ValueError('One-GPU ordering requires all30 paired fits complete')
    from . import frozen_heads as H, export_features as E
    completed = []
    for condition in tuple(method + '-frozen-smallhead' for method in METHODS) + CONTROLS:
        S.save(S.OUT / 'ready_queue/status.json', {'status': 'running', 'condition': condition,
               'completed_conditions': completed, 'updated_at': S.now(), 'test_evaluation_frozen': True})
        try:
            if condition not in H.REUSE_CONDITIONS:
                E.export(condition.removesuffix('-frozen-smallhead'), batch_size=4)
            H.run((condition,), resume=resume)
        except BaseException as error:
            S.save(S.OUT / 'ready_queue/status.json', {'status': 'failed', 'condition': condition,
                   'completed_conditions': completed, 'error_type': type(error).__name__, 'error': str(error),
                   'updated_at': S.now(), 'test_evaluation_frozen': True})
            raise
        completed.append(condition)
    S.save(S.OUT / 'ready_queue/status.json', {'status': 'registered_ready_batch_complete',
           'completed_conditions': completed, 'formal_fits_in_this_batch': 135,
           'full360_plan_complete': False, 'test_evaluation_frozen': True, 'updated_at': S.now()})
    print('Ready frozen batch complete:135 fits; remaining registered routes still required before external tests.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-paired', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(wait_paired=args.wait_paired, resume=args.resume)
