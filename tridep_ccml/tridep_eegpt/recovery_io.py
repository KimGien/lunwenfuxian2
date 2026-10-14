"""Bounded file replacement recovery, without retrying model computation.

Only the I/O functions of the frozen producer are bridged in memory.  A
serialized temporary file is written once; a Windows sharing/permission
failure can retry only its atomic replacement.  Any unresolved temporary
file is retained as evidence and cannot be overwritten by a later save.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import time

from . import common as C
from . import train_eval as old

MAX_REPLACE_ATTEMPTS = 40
MAX_REPLACE_SECONDS = 5.0
MAX_SLEEP_SECONDS = 0.1
_installed = False


def _inside(path, roots):
    path = Path(path).resolve()
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("Recovery I/O path is outside OUT/REPORT: " + str(path))
    return path


def _sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _same(left, right):
    """Compare immutable serialized values without constructing a model."""
    if C.torch.is_tensor(left) or C.torch.is_tensor(right):
        return (C.torch.is_tensor(left) and C.torch.is_tensor(right)
                and left.dtype == right.dtype and left.shape == right.shape
                and C.torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, C.np.ndarray) or isinstance(right, C.np.ndarray):
        return (isinstance(left, C.np.ndarray) and isinstance(right, C.np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and C.np.array_equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def installed_io(event_log_name="io_events.jsonl"):
    """Temporarily install the three serialization-compatible I/O bridges."""
    global _installed
    if _installed:
        raise RuntimeError("Recovery I/O bridge is already installed")
    if (not isinstance(event_log_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.jsonl", event_log_name) is None
            or event_log_name.split(".", 1)[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)),
             *("LPT" + str(i) for i in range(1, 10))}):
        raise ValueError("Recovery event log requires an ordinary .jsonl basename")
    roots = (C.OUT.resolve(), C.REPORT.resolve())
    recovery_root = _inside(C.OUT / "recovery", (roots[0],))
    event_path = _inside(recovery_root / event_log_name, (recovery_root,))
    if event_path != recovery_root / event_log_name:
        raise ValueError("Recovery event log basename is redirected")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    original = (C.save, old._save_torch, old._save_array)

    def event(kind, **fields):
        # Deliberately append directly: never call any bridged save function.
        payload = {"time": C.now(), "kind": kind, **fields}
        with event_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def paths(path):
        path = _inside(path, roots)
        temporary = _inside(path.with_suffix(path.suffix + ".part"), roots)
        if temporary != path.with_suffix(path.suffix + ".part"):
            raise ValueError("Recovery temporary path is redirected: " + str(temporary))
        if temporary.exists():
            raise ValueError("Preserve existing recovery temporary artifact: " + str(temporary))
        return path, temporary

    def replace(temporary, path):
        started = time.monotonic()
        temporary_sha256 = None
        for attempt in range(1, MAX_REPLACE_ATTEMPTS + 1):
            try:
                temporary.replace(path)
            except OSError as error:
                eligible = isinstance(error, PermissionError) or getattr(error, "winerror", None) == 5
                elapsed = time.monotonic() - started
                if temporary_sha256 is None and temporary.exists():
                    temporary_sha256 = _sha256(temporary)
                remaining = MAX_REPLACE_SECONDS - elapsed
                retry = eligible and attempt < MAX_REPLACE_ATTEMPTS and remaining > 0
                event("replace_retry" if retry else "replace_failed", path=str(path),
                      temporary=str(temporary), temporary_sha256=temporary_sha256,
                      attempt=attempt, elapsed_seconds=elapsed,
                      error_type=type(error).__name__, error=str(error),
                      errno=error.errno, winerror=getattr(error, "winerror", None))
                if not retry:
                    raise
                # No serialization or model operation occurs inside this loop.
                time.sleep(min(MAX_SLEEP_SECONDS, remaining))
                if time.monotonic() - started >= MAX_REPLACE_SECONDS:
                    event("replace_failed", path=str(path), temporary=str(temporary),
                          temporary_sha256=temporary_sha256, attempt=attempt,
                          elapsed_seconds=time.monotonic() - started,
                          error_type=type(error).__name__, error=str(error),
                          errno=error.errno, winerror=getattr(error, "winerror", None))
                    raise
            else:
                if attempt > 1:
                    event("replace_recovered", path=str(path), temporary=str(temporary),
                          temporary_sha256=temporary_sha256, attempt=attempt,
                          elapsed_seconds=time.monotonic() - started)
                return

    def save(path, data, immutable=False):
        path = _inside(path, roots)
        if immutable and path.exists():
            if C.read(path) != data:
                raise ValueError("Existing frozen record differs: " + str(path))
            event("immutable_reused", path=str(path), format="json")
            return
        path, temporary = paths(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
                             + "\n", encoding="utf-8")
        replace(temporary, path)

    def save_torch(path, value):
        path = _inside(path, roots)
        value = old._cpu(value)
        if path.exists():
            saved = C.torch.load(path, map_location="cpu", weights_only=False)
            if not _same(saved, value):
                raise ValueError("Existing frozen checkpoint differs: " + str(path))
            event("immutable_reused", path=str(path), format="torch")
            return
        path, temporary = paths(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        C.torch.save(value, temporary)
        replace(temporary, path)

    def save_array(path, values, compressed=False):
        path = _inside(path, roots)
        if path.exists():
            if compressed:
                with C.np.load(path, allow_pickle=False) as saved:
                    same = set(saved.files) == set(values) and all(_same(saved[k], values[k]) for k in saved.files)
            else:
                same = _same(C.np.load(path, allow_pickle=False), values)
            if not same:
                raise ValueError("Existing frozen array artifact differs: " + str(path))
            event("immutable_reused", path=str(path), format="npz" if compressed else "npy")
            return
        path, temporary = paths(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            if compressed:
                C.np.savez_compressed(stream, **values)
            else:
                C.np.save(stream, values, allow_pickle=False)
        replace(temporary, path)

    event("bridge_installed", max_attempts=MAX_REPLACE_ATTEMPTS,
          max_seconds=MAX_REPLACE_SECONDS, max_sleep_seconds=MAX_SLEEP_SECONDS,
          retries_cover_model_computation=False)
    _installed = True
    C.save, old._save_torch, old._save_array = save, save_torch, save_array
    try:
        yield
    finally:
        C.save, old._save_torch, old._save_array = original
        _installed = False
        event("bridge_removed")
