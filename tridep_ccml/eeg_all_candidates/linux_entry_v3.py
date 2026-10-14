"""Qualified isolated Linux runtime with the same train and test mathematics."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys
from . import linux_entry as L1, linux_entry_v2 as L2, storage as S

ACTIONS = L2.ACTIONS + ('mamba-acceleration-prepare',)


def execute(action, *, resume=False, owner_record=None, acceleration_registration=None):
    if action not in ACTIONS:
        raise ValueError('Unknown Linux action')
    if action not in ('mamba-acceleration-prepare', 'mamba-finetune', 'mamba-evaluate'):
        return L2.execute(action, resume=resume, owner_record=owner_record)
    owner = L1._register_owner(action, owner_record)
    if action == 'mamba-acceleration-prepare':
        if resume:
            raise ValueError('Preparation does not resume formal optimization')
        if sys.platform != 'linux' or Path(sys.prefix).resolve() != (S.OUT / 'assets/eegmamba_linux_env').resolve():
            raise ValueError('Preparation requires the isolated official Linux runtime')
        for package, version in {'torch': '2.5.1+cu124', 'mamba_ssm': '2.2.4', 'causal_conv1d': '1.5.0.post8', 'triton': '3.1.0'}.items():
            if importlib.metadata.version(package) != version:
                raise ValueError('Preparation runtime package differs: ' + package)
        from . import acceleration_prepare_v1 as A
        return A.run('EEGMamba-lastblock-finetune', linux_owner=S.desc(owner))
    L1._require_runtime()
    if action == 'mamba-evaluate':
        if resume:
            raise ValueError('Evaluation uses the original immutable probability cache')
        from . import evaluate_v3 as E
        return E.run(('EEGMamba-lastblock-finetune',), no_report=True)
    if acceleration_registration is None:
        raise ValueError('Linux FT requires an immutable qualified execution registration')
    from . import finetune_v3 as F
    return F.run(('EEGMamba-lastblock-finetune',), acceleration_registration=acceleration_registration,
                 resume=resume, micro_subjects=1, window_chunk=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=ACTIONS, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--owner-record', required=True)
    parser.add_argument('--acceleration-registration')
    parser.add_argument('--acceleration-registration-sha256')
    args = parser.parse_args()
    if bool(args.acceleration_registration) != bool(args.acceleration_registration_sha256):
        parser.error('Execution registration path and SHA must both be provided')
    binding = {'path': args.acceleration_registration, 'sha256': args.acceleration_registration_sha256} if args.acceleration_registration else None
    execute(args.action, resume=args.resume, owner_record=args.owner_record, acceleration_registration=binding)
