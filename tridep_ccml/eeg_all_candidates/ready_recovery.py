"""Registered CodeBrain cache initialization repair, without editing old sources.

The original failed export and all its partial artifacts remain intact. A
separate versioned export binds the weight-only author buffer initialization;
its completed manifest is then the frozen-head condition's explicit input.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import storage as S, registry as R


def repair():
    from . import export_features as E
    from .adapters import codebrain_v2 as adapter
    R.load_registration()
    condition = 'CodeBrain-frozen-smallhead'
    main_root = S.OUT
    destination = main_root / 'frozen_features' / condition / 'manifest.json'
    failure = S.read(main_root / 'ready_queue/status.json')
    if failure.get('status') != 'failed' or failure.get('condition') != condition:
        raise ValueError('Only the actual recorded CodeBrain frozen-export failure may enter this repair')
    registration = main_root / 'ready_recovery/registration.json'
    if not registration.exists():
        original = main_root / 'frozen_features' / condition
        original_artifacts = [S.desc(path) for path in sorted(original.rglob('*')) if path.is_file()]
        S.save(registration, {'registry': R.registration_descriptor(), 'registered_at': S.now(),
                             'failed_queue_status': failure,
                             'sources': [S.desc(Path(__file__)), S.desc(adapter.__file__), S.desc(E.__file__)],
                             'versioned_export': 'ready_recovery/codebrain_v2_export',
                             'original_sources_and_partial_export_preserved': True,
                             'original_partial_export_artifacts': original_artifacts,
                             'test_forward_allowed': False, 'encoder_fit_subject_ids': []}, immutable=True)
    for source in S.read(registration)['sources']:
        S.checked(source)
    for source in S.read(registration)['original_partial_export_artifacts']:
        S.checked(source)
    if destination.exists():
        manifest = S.read(destination)
        if manifest.get('repair_registration') != S.desc(registration):
            raise ValueError('Main condition already has an unrelated feature manifest')
        for source in manifest['sources']:
            S.checked(source)
        return manifest
    key = 'tridep_ccml.eeg_all_candidates.adapters.codebrain'
    previous = sys.modules.get(key)
    # The untouched generic exporter records adapter.__file__; the new source
    # is therefore explicit in registration, sidecars and final provenance.
    sys.modules[key] = adapter
    try:
        S.OUT = main_root / 'ready_recovery/codebrain_v2_export'
        manifest = E.export('CodeBrain', batch_size=4)
    finally:
        S.OUT = main_root
        if previous is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous
    manifest['repair_registration'] = S.desc(registration)
    manifest['original_export_preserved'] = True
    manifest['sources'].append(S.desc(registration))
    for source in S.read(registration)['original_partial_export_artifacts']:
        S.checked(source)
    S.save(destination, manifest, immutable=True)
    print('CodeBrain versioned frozen export complete; original failed attempt preserved.', flush=True)
    return manifest


if __name__ == '__main__':
    repair()
