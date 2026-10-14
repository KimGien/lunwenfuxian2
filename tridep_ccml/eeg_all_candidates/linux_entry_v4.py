"""Q5 Linux entry; original qualified F3 train/E3 evaluation are unchanged."""
import argparse
import importlib.metadata
from pathlib import Path
import sys
from . import linux_entry as L1, linux_entry_v3 as L3, storage as S

ACTIONS = L3.ACTIONS


def execute(action, *, resume=False, owner_record=None, acceleration_registration=None):
    if action != 'mamba-acceleration-prepare':
        return L3.execute(action, resume=resume, owner_record=owner_record, acceleration_registration=acceleration_registration)
    if resume: raise ValueError('Preparation does not resume formal optimization')
    owner = L1._register_owner(action, owner_record)
    if sys.platform != 'linux' or Path(sys.prefix).resolve() != (S.OUT / 'assets/eegmamba_linux_env').resolve():
        raise ValueError('Original isolated official Linux runtime required')
    for name, version in {'torch': '2.5.1+cu124', 'mamba_ssm': '2.2.4', 'causal_conv1d': '1.5.0.post8', 'triton': '3.1.0'}.items():
        if importlib.metadata.version(name) != version: raise ValueError('Package version differs: ' + name)
    from . import acceleration_prepare_v2 as A
    return A.run('EEGMamba-lastblock-finetune', linux_owner=S.desc(owner))


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
