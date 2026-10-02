"""Shared verified EEG views for the three downstream comparison tracks."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from tridep_ccml.v2.common import WORK, TC, descriptor, verify_descriptor, read_json, require

OUT = TC / 'experiments/EEG-AUG-JITTER-37-fulltext'
BASE_EEG = TC / 'experiments/CCML-v2-E-37-fulltext'
VIEWS = ('clean', 'jitter_a', 'jitter_b')


def load_views() -> dict[str, np.ndarray]:
    """Return [clean, jitter_a, jitter_b] x [60,200] per subject.

    Every view is label-free. Callers must index augmentation views only for
    subjects in their own fold's training list.
    """
    rows = read_json(OUT / 'feature_manifest.json')
    require(len(rows) == 37 and len({row['subject_id'] for row in rows}) == 37,
            'Incomplete augmentation cohort')
    result: dict[str, np.ndarray] = {}
    for row in rows:
        require(row['views'] == list(VIEWS) and row['fit_subject_ids'] == []
                and row['labels_used'] is False, 'Augmentation provenance mismatch')
        arrays = [np.load(verify_descriptor(row['features'][v]), allow_pickle=False) for v in VIEWS]
        require(all(a.shape == (60, 200) and a.dtype == np.float32 and np.isfinite(a).all()
                    for a in arrays), 'Bad EEG augmented feature')
        old = np.load(verify_descriptor(row['clean_source']), allow_pickle=False)
        require(np.array_equal(arrays[0], old), 'Clean view changed')
        result[row['subject_id']] = np.stack(arrays, axis=0)
    return result


def chosen_view(seed: int, fold: int, epoch: int, sid: str) -> int:
    require(seed in (11, 22, 33) and fold in range(1, 6) and epoch in range(1, 61),
            'Unexpected training view request')
    key = f'eeg-aug-jitter-v1|{seed}|{fold}|{epoch}|{sid}'.encode()
    return int.from_bytes(sha256(key).digest()[:8], 'little') % 3


def epoch_views(ids: list[str], views: dict[str, np.ndarray], seed: int,
                fold: int, epoch: int) -> np.ndarray:
    """One view per training person; no RNG state is mutated."""
    require(len(ids) == len(set(ids)) and set(ids) <= set(views), 'Invalid train IDs')
    return np.stack([views[sid][chosen_view(seed, fold, epoch, sid)] for sid in ids]).astype(np.float32)


def digest_file(path: Path) -> str:
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
