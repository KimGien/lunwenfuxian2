"""Durable experiment files; replacement retries never repeat model work."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def desc(path):
    return {'path': Path(path).resolve().relative_to(WORK).as_posix(), 'sha256': digest(path)}


def checked(record):
    path = (WORK / record['path']).resolve()
    if not path.is_relative_to(WORK) or digest(path) != record['sha256']:
        raise ValueError('Source changed: ' + str(path))
    return path


def write(path, serialize, *, immutable=False):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (OUT, REPORT)):
        raise ValueError('Experiment output outside authorized experiment directories')
    if immutable and path.exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    with temporary.open('wb') as stream:
        serialize(stream)
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            eligible = isinstance(error, PermissionError) or getattr(error, 'winerror', None) == 5
            if not eligible or attempt == 39 or time.monotonic() - started >= 5:
                raise
            time.sleep(.1)


def save(path, value, *, immutable=False):
    if immutable and Path(path).exists():
        if read(path) != value:
            raise ValueError('Existing immutable JSON differs: ' + str(path))
        return
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    write(path, lambda stream: stream.write(raw), immutable=immutable)


def save_torch(path, value, *, immutable=False):
    import torch
    if immutable and Path(path).exists():
        from tridep_ccml.tridep_eegpt.recovery_io import _same
        existing = torch.load(path, map_location='cpu', weights_only=False)
        if not _same(existing, value):
            raise ValueError('Existing immutable checkpoint differs: ' + str(path))
        return
    write(path, lambda stream: torch.save(value, stream), immutable=immutable)
