"""Qualified Linux entry; sealed owner/check/export/evaluation behavior is reused."""
from __future__ import annotations

import argparse

from . import linux_entry as V1

ACTIONS = V1.ACTIONS


def execute(action, *, resume=False, owner_record=None):
    if action not in ACTIONS:
        raise ValueError('Unknown qualified Linux action')
    if action not in ('mamba-finetune-prepare', 'mamba-finetune', 'mamba-evaluate'):
        return V1.execute(action, resume=resume, owner_record=owner_record)
    V1._register_owner(action, owner_record)
    V1._require_runtime()
    if action == 'mamba-evaluate':
        if resume:
            raise ValueError('Evaluation uses bound probability-cache resume without --resume')
        from . import evaluate_v2 as E
        return E.run(('EEGMamba-lastblock-finetune',), no_report=True)
    from . import finetune_v2 as F
    if action == 'mamba-finetune-prepare':
        if resume:
            raise ValueError('--resume is only for formal Mamba fitting')
        return F.prepare_condition('EEGMamba-lastblock-finetune', micro_subjects=1, window_chunk=8)
    return F.run(('EEGMamba-lastblock-finetune',), resume=resume, micro_subjects=1, window_chunk=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=ACTIONS, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--owner-record', required=True)
    args = parser.parse_args()
    execute(args.action, resume=args.resume, owner_record=args.owner_record)
