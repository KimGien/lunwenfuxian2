"""Versioned execution-only SSL precision correction; the360-fit protocol stays fixed."""
from __future__ import annotations

import argparse
from pathlib import Path
import types

from . import continuation_queue as V1, registry as R, storage as S

ROOT = S.OUT / 'continuation_queue_v2'
OLD_ROOT = V1.ROOT
SOURCE = Path(__file__).resolve()
NEW_SOURCES = ('continuation_queue_v2.py', 'handoff_supervisor_v2.py', 'test_continuation_queue_v2.py',
               'ssl_v2.py', 'test_ssl_v2.py')


def jobs():
    values = [{**job, 'module': 'ssl_v2' if job['module'] == 'ssl' else job['module']} for job in V1.jobs()]
    original = V1.jobs()
    if len(values) != len(original) or sum(job.get('fits', 0) for job in values) != 195:
        raise ValueError('V2 must retain the exact195 registered jobs')
    for before, after in zip(original, values):
        restored = dict(after)
        if restored['module'] == 'ssl_v2':
            restored['module'] = 'ssl'
        if restored != before:
            raise ValueError('Only SSL execution module may change')
    return values


def sources():
    if not (OLD_ROOT / 'registration.json').exists():
        raise ValueError('Frozen original queue registration required')
    old = S.read(OLD_ROOT / 'registration.json')
    for source in old['sources']:
        S.checked(source)
    paths = [SOURCE.parent / name for name in NEW_SOURCES] + [OLD_ROOT / 'registration.json']
    records = list(old['sources']) + [S.desc(path) for path in paths]
    return list({record['path']: record for record in records}.values())


def register():
    R.load_registration()
    path = ROOT / 'registration.json'
    if not path.exists():
        S.save(path, {'version': 2, 'registry': R.registration_descriptor(), 'registered_at': S.now(),
                     'jobs': jobs(), 'sources': sources(), 'old_execution_registration': S.desc(OLD_ROOT / 'registration.json'),
                     'one_gpu_sequential': True, 'formal_fits_in_reused_matrix': 195,
                     'change': 'only ssl.py dispatch becomes ssl_v2.py; complete SSL loss autocast disabled as originally intended FP32',
                     'original_protocol_or_training_hyperparameters_changed': False,
                     'conformer_completed_fits_reused': 15, 'new_remaining_fits_after_handoff': 180,
                     'requires_handoff_complete_before_any_child': True, 'test_gate_unchanged': True}, immutable=True)
    value = S.read(path)
    if value['jobs'] != jobs() or value['registry'] != R.registration_descriptor():
        raise ValueError('V2 execution registration changed')
    for source in value['sources']:
        S.checked(source)
    return value


def require_handoff():
    value = S.read(ROOT / 'handoff_complete.json')
    if not (value['passed'] is True and value['old_dispatcher_terminated_after_children'] is True
            and value['launcher_exit_code'] == value['gpu_child_exit_code'] == 0
            and value['conformer_fits_verified'] == 15 and value['ssl_v1_never_started'] is True
            and value['v2_registration'] == S.desc(ROOT / 'registration.json')):
        raise ValueError('Exact safe handoff proof required before v2 starts')
    for record in value['protected_sources'] + value['conformer_completed_artifacts']:
        S.checked(record)
    if not V1.all_complete(('EEGConformer-scratch',)) or not V1.all_complete(V1.FIRST):
        raise ValueError('Completed FIRST165 and Conformer15 must verify again')
    return value


def isolated_driver():
    """Copy frozen function definitions into separate globals; never mutate V1."""
    module = types.ModuleType(__package__ + '._continuation_v2_runtime')
    module.__package__ = __package__
    module.__file__ = V1.__file__
    exec(compile(Path(V1.__file__).read_text(encoding='utf-8'), V1.__file__, 'exec'), module.__dict__)
    module.ROOT = ROOT
    module.jobs = jobs
    module.sources = sources
    module.register = register
    return module


def run(*, resume=True):
    register()
    require_handoff()
    # Existing Conformer prepare/fits are hash-verified no-ops on --resume.
    isolated_driver().run(resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    register() if args.register_only else run(resume=args.resume)
