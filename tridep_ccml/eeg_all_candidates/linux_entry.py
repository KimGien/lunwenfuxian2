"""Explicit official Linux Mamba actions for the single-GPU continuation queue."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import uuid

ACTIONS = ('mamba-check', 'mamba-export', 'mamba-finetune-prepare', 'mamba-finetune', 'mamba-evaluate')


def _require_runtime():
    from . import storage as S
    expected = S.OUT / 'assets/eegmamba_linux_env'
    if sys.platform != 'linux' or Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError('This action requires the separately qualified official workspace Linux Mamba environment')
    import importlib.metadata
    import torch
    if torch.__version__ != '2.5.1+cu124' or importlib.metadata.version('mamba_ssm') != '2.2.4':
        raise RuntimeError('Linux Mamba runtime version differs from qualified binding')
    return S


def _check_with_attempt_ledger(S):
    """Observe real check calls, including failures; never count a formal fit."""
    import torch
    from . import check_eegmamba_gpu as check
    path = S.OUT / 'engineering_attempts/EEGMamba' / (str(time.time_ns()) + '_' + uuid.uuid4().hex + '.json')
    record = {'action': 'mamba-check', 'started_at': S.now(), 'finished_at': None, 'status': 'running',
              'source': S.desc(Path(__file__)), 'check_source': S.desc(Path(check.__file__)),
              'encode_started': 0, 'encode_returned': 0,
              'backward_started': 0, 'backward_returned': 0,
              'sgd_step_started': 0, 'sgd_step_returned': 0,
              'formal_fits': 0, 'formal_optimizer_updates': 0,
              'cohort_data_or_labels_used': False, 'engineering': True}
    S.save(path, record, immutable=True)

    def observe(kind, fn):
        def wrapped(*args, **kwargs):
            record[kind + '_started'] += 1
            S.save(path, record)
            value = fn(*args, **kwargs)
            record[kind + '_returned'] += 1
            S.save(path, record)
            return value
        return wrapped

    encode = check.eegmamba.encode_windows
    backward = torch.Tensor.backward
    sgd_step = torch.optim.SGD.step
    previous_argv = sys.argv
    try:
        check.eegmamba.encode_windows = observe('encode', encode)
        torch.Tensor.backward = observe('backward', backward)
        torch.optim.SGD.step = observe('sgd_step', sgd_step)
        sys.argv = [str(check.__file__), '--run']
        check.main()
        record.update(status='complete', passed=True)
    except BaseException as exc:
        record.update(status='failed', passed=False, failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        check.eegmamba.encode_windows = encode
        torch.Tensor.backward = backward
        torch.optim.SGD.step = sgd_step
        sys.argv = previous_argv
        record['finished_at'] = S.now()
        S.save(path, record)


def _register_owner(action, relative_path=None):
    """Expose the actual Linux GPU-process group before importing its runtime."""
    from . import storage as S
    if sys.platform != 'linux':
        raise RuntimeError('Linux actions require a Linux process')
    if os.getpgrp() != os.getpid():
        os.setsid()
    root = (S.OUT / 'continuation_queue/attempts').resolve()
    if relative_path is None:
        path = root / ('linux_owner_' + str(time.time_ns()) + '_' + uuid.uuid4().hex + '.json')
    else:
        value = Path(relative_path)
        if value.is_absolute():
            raise ValueError('--owner-record must be a relative experiment path')
        # Support queue paths relative either to the experiment or workspace.
        path = ((S.WORK if value.parts and value.parts[0] == 'tridep_ccml' else S.OUT) / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError('--owner-record must stay inside OUT/continuation_queue/attempts')
    S.save(path, {'pid': os.getpid(), 'pgid': os.getpgrp(), 'started_at': S.now(), 'action': action}, immutable=True)
    return path


def execute(action, *, resume=False, owner_record=None):
    if action not in ACTIONS:
        raise ValueError('Unknown Linux action')
    # The owner file must be durable before runtime qualification or GPU work.
    _register_owner(action, owner_record)
    S = _require_runtime()
    if action == 'mamba-check':
        if resume:
            raise ValueError('--resume is only for formal mamba-finetune')
        return _check_with_attempt_ledger(S)
    if action == 'mamba-export':
        if resume:
            raise ValueError('Frozen export resumes via its immutable manifest, without --resume')
        from . import export_features as E
        # The immutable exporter inserts the Windows wheel target. Linux torch
        # and NumPy have already loaded from the qualified environment; remove
        # that target before dynamically importing the official Mamba adapter.
        sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != E.DEPS.resolve()]
        return E.export('EEGMamba', batch_size=4)
    if action == 'mamba-evaluate':
        if resume:
            raise ValueError('Evaluation resumes durable probability batches without --resume')
        from . import evaluate as E
        return E.run(('EEGMamba-lastblock-finetune',), no_report=True)
    from . import finetune as F
    if action == 'mamba-finetune-prepare':
        if resume:
            raise ValueError('--resume is only for formal mamba-finetune')
        return F.prepare_condition('EEGMamba-lastblock-finetune', micro_subjects=1, window_chunk=8)
    return F.run(('EEGMamba-lastblock-finetune',), resume=resume, micro_subjects=1, window_chunk=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=ACTIONS, required=True)
    parser.add_argument('--resume', action='store_true', help='Full-state resume for formal mamba-finetune only')
    parser.add_argument('--owner-record', help='Relative owner JSON path within OUT/continuation_queue/attempts')
    args = parser.parse_args()
    execute(args.action, resume=args.resume, owner_record=args.owner_record)


if __name__ == '__main__':
    main()
