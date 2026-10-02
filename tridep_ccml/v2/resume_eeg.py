"""Resume the pinned EEG runner with a narrowly scoped TorchVersion loader.

The runner and first trained checkpoint are preserved byte-for-byte. PyTorch's
string-like version metadata needs one additional safe type with weights_only.
This entry point is required for this version's checkpoints and records itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from .common import (OUT, WORK, ADAPT, descriptor, verify_descriptor, read_json,
                     require, write_new, check_existing)
import torch


@contextmanager
def version_metadata_loader():
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        yield


def load_checkpoint(path):
    """Only load a hash-verified experiment checkpoint, keeping weights_only."""
    require(path.resolve().is_relative_to(OUT.resolve()), 'Checkpoint outside this experiment')
    side = read_json(path.with_suffix('.json'))
    require(verify_descriptor(side['checkpoint']).resolve() == path.resolve(), 'Checkpoint path mismatch')
    with version_metadata_loader():
        saved = torch.load(path, map_location='cpu', weights_only=True)
    require(saved['metadata'] == side['metadata'], 'Checkpoint metadata mismatch')
    require(all(torch.is_tensor(v) and torch.isfinite(v).all() for v in saved['state_dict'].values()),
            'Invalid checkpoint tensors')
    return saved


def run():
    from .rebuild_eeg import run as pinned_run
    before = check_existing()
    first = OUT/'runs'/ADAPT/'seed_11/fold_1/checkpoint.pt'
    saved = load_checkpoint(first)
    require(saved['metadata']['completed_epochs'] == 60 and saved['metadata']['optimizer_updates'] == 240,
            'First fit was not complete')
    history = read_json(first.parent/'train_history.json')
    require([r['epoch'] for r in history] == list(range(1,61)), 'Incomplete first fit history')
    verify_descriptor(saved['metadata']['runner'])
    write_new(OUT/'serialization_recovery.json', {
        'checkpoint': descriptor(first), 'checkpoint_not_rewritten': True,
        'training_not_repeated_for_first_fold': True, 'loader': descriptor(__file__),
        'pinned_runner': saved['metadata']['runner'], 'weights_only': True,
        'additional_safe_type': 'torch.torch_version.TorchVersion',
        'protected_files_verified_before_resume': before,
        'note': 'Allow PyTorch string-like version metadata; no unrestricted pickle loading.'})
    # The unchanged runner also reloads all newly trained checkpoints this way.
    with version_metadata_loader():
        pinned_run()


if __name__ == '__main__':
    run()
