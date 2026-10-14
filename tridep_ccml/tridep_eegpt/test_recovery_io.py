"""Synthetic file-I/O checks; no model construction, forward, or fitting."""
from __future__ import annotations

import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import common as C
from . import train_eval as old
from . import recovery_io as R


class RecoveryIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.out = self.base / "out"
        self.report = self.base / "report"
        self.original = (C.save, old._save_torch, old._save_array)
        self.out_patch = patch.object(C, "OUT", self.out)
        self.report_patch = patch.object(C, "REPORT", self.report)
        self.out_patch.start(); self.report_patch.start()

    def tearDown(self):
        self.out_patch.stop(); self.report_patch.stop()
        self.temporary.cleanup()

    def events(self):
        return [json.loads(line) for line in (self.out / "recovery/io_events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_formats_immutable_reuse_and_restoration(self):
        record = {"汉字": [1, 2], "truth": True}
        tensor = {"state_dict": {"synthetic": C.torch.tensor([1., 2.])}, "metadata": record}
        array = C.np.arange(6, dtype=C.np.int64).reshape(2, 3)
        bundle = {"values": array, "ids": C.np.array(["synthetic"])}
        with R.installed_io():
            C.save(self.out / "record.json", record, immutable=True)
            self.assertEqual((self.out / "record.json").read_text(encoding="utf-8"),
                             json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            C.save(self.out / "record.json", record, immutable=True)
            with self.assertRaises(ValueError):
                C.save(self.out / "record.json", {"different": True}, immutable=True)
            old._save_torch(self.out / "initial.pt", tensor)
            old._save_torch(self.out / "initial.pt", tensor)
            with self.assertRaises(ValueError):
                old._save_torch(self.out / "initial.pt", {"different": True})
            old._save_array(self.out / "order.npy", array)
            old._save_array(self.out / "order.npy", array)
            with self.assertRaises(ValueError):
                old._save_array(self.out / "order.npy", array + 1)
            old._save_array(self.report / "bundle.npz", bundle, compressed=True)
            old._save_array(self.report / "bundle.npz", bundle, compressed=True)
            with self.assertRaises(ValueError):
                old._save_array(self.report / "bundle.npz", {"values": array + 1}, compressed=True)
            with C.np.load(self.report / "bundle.npz", allow_pickle=False) as data:
                self.assertTrue(C.np.array_equal(data["values"], array))
                self.assertEqual(set(data.files), set(bundle))
        self.assertEqual((C.save, old._save_torch, old._save_array), self.original)
        self.assertEqual(sum(e["kind"] == "immutable_reused" for e in self.events()), 4)

    def test_retry_replaces_same_bytes_without_reserialization(self):
        original_replace = Path.replace
        attempted = []
        def sharing_failure(path, target):
            attempted.append((path, path.read_bytes()))
            if len(attempted) <= 2:
                raise PermissionError(errno.EACCES, "synthetic sharing conflict")
            return original_replace(path, target)
        with R.installed_io(), patch.object(Path, "replace", sharing_failure), patch.object(R.time, "sleep") as sleep:
            C.save(self.out / "progress.json", {"updates_completed": 82})
        self.assertEqual(len(attempted), 3)
        self.assertTrue(all(x == attempted[0] for x in attempted))
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(call.args[0] <= .1 for call in sleep.call_args_list))
        self.assertEqual([e["kind"] for e in self.events()],
                         ["bridge_installed", "replace_retry", "replace_retry", "replace_recovered", "bridge_removed"])

    def test_persistent_permission_error_is_bounded_and_preserves_temporary(self):
        with R.installed_io(), patch.object(Path, "replace", side_effect=PermissionError("persistent")) as replace, \
                patch.object(R.time, "sleep"):
            with self.assertRaises(PermissionError):
                C.save(self.out / "progress.json", {"updates_completed": 82})
            self.assertEqual(replace.call_count, 40)
            temp = self.out / "progress.json.part"
            self.assertTrue(temp.exists())
            original_bytes = temp.read_bytes()
            with self.assertRaises(ValueError):
                C.save(self.out / "progress.json", {"updates_completed": 83})
            self.assertEqual(temp.read_bytes(), original_bytes)
        self.assertEqual(self.events()[-2]["kind"], "replace_failed")

    def test_winerror5_retries_each_binary_format_without_reserialization(self):
        original_replace = Path.replace
        attempts = {}
        written = {}
        def windows_failure(path, target):
            attempts[path] = attempts.get(path, 0) + 1
            if attempts[path] == 1:
                written[path] = path.read_bytes()
                failure = OSError("synthetic Windows access denied")
                failure.winerror = 5
                raise failure
            self.assertEqual(path.read_bytes(), written[path])
            return original_replace(path, target)
        with R.installed_io(), patch.object(Path, "replace", windows_failure), patch.object(R.time, "sleep"), \
                patch.object(C.torch, "save", wraps=C.torch.save) as torch_save, \
                patch.object(C.np, "save", wraps=C.np.save) as np_save, \
                patch.object(C.np, "savez_compressed", wraps=C.np.savez_compressed) as np_savez:
            old._save_torch(self.out / "initial.pt", {"synthetic": C.torch.tensor([1.])})
            old._save_array(self.out / "order.npy", C.np.array([1]))
            old._save_array(self.out / "bundle.npz", {"synthetic": C.np.array([1])}, compressed=True)
            self.assertEqual(torch_save.call_count, 1)
            self.assertEqual(np_save.call_count, 1)
            self.assertEqual(np_savez.call_count, 1)
        self.assertEqual(sorted(attempts.values()), [2, 2, 2])
        self.assertEqual(sum(e["kind"] == "replace_recovered" for e in self.events()), 3)

    def test_unknown_error_is_never_retried(self):
        with R.installed_io(), patch.object(Path, "replace", side_effect=OSError(errno.EIO, "synthetic I/O failure")) as replace, \
                patch.object(R.time, "sleep") as sleep:
            with self.assertRaises(OSError):
                C.save(self.out / "record.json", {"value": 1})
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(sleep.call_count, 0)
            self.assertTrue((self.out / "record.json.part").exists())

    def test_deadline_is_bounded(self):
        with R.installed_io(), patch.object(Path, "replace", side_effect=PermissionError("persistent")) as replace, \
                patch.object(R.time, "monotonic", side_effect=[0., 0., 5., 5.]), patch.object(R.time, "sleep"):
            with self.assertRaises(PermissionError):
                C.save(self.out / "record.json", {"value": 1})
            self.assertEqual(replace.call_count, 1)
        self.assertTrue((self.out / "record.json.part").exists())

    def test_outside_paths_rejected_and_existing_temporary_preserved(self):
        self.out.mkdir()
        temporary = self.out / "record.json.part"
        temporary.write_bytes(b"old failure evidence")
        with R.installed_io():
            for writer, values in ((C.save, {}), (old._save_torch, {}), (old._save_array, C.np.array([1]))):
                with self.assertRaises(ValueError):
                    writer(self.base / "outside", values)
            with self.assertRaises(ValueError):
                C.save(self.out / "record.json", {})
            self.assertEqual(temporary.read_bytes(), b"old failure evidence")
            with self.assertRaises(RuntimeError):
                with R.installed_io():
                    pass

    def test_error_restores_functions_and_append_only_events(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic caller failure"):
            with R.installed_io():
                raise RuntimeError("synthetic caller failure")
        old_bytes = (self.out / "recovery/io_events.jsonl").read_bytes()
        with R.installed_io():
            C.save(self.out / "mutable.json", {"value": 1})
            C.save(self.out / "mutable.json", {"value": 2})
        self.assertTrue((self.out / "recovery/io_events.jsonl").read_bytes().startswith(old_bytes))
        self.assertEqual(C.read(self.out / "mutable.json"), {"value": 2})
        self.assertEqual((C.save, old._save_torch, old._save_array), self.original)

    def test_custom_event_basename_keeps_original_log_unchanged(self):
        with R.installed_io():
            C.save(self.out / "train_record.json", {"synthetic": True})
        original_bytes = (self.out / "recovery/io_events.jsonl").read_bytes()
        with R.installed_io(event_log_name="finalization_io_events.jsonl"):
            C.save(self.report / "final_record.json", {"synthetic": True})
        self.assertEqual((self.out / "recovery/io_events.jsonl").read_bytes(), original_bytes)
        finalization = self.out / "recovery/finalization_io_events.jsonl"
        old_bytes = finalization.read_bytes()
        records = [json.loads(line) for line in old_bytes.decode("utf-8").splitlines()]
        self.assertEqual([r["kind"] for r in records], ["bridge_installed", "bridge_removed"])
        with R.installed_io(event_log_name="finalization_io_events.jsonl"):
            pass
        self.assertTrue(finalization.read_bytes().startswith(old_bytes))
        self.assertEqual((self.out / "recovery/io_events.jsonl").read_bytes(), original_bytes)

    def test_custom_event_name_rejects_paths_and_windows_devices(self):
        for name in ("", ".", "..", "../outside.jsonl", "sub/events.jsonl", "sub\\events.jsonl",
                     "C:\\events.jsonl", "events.jsonl:stream", "events.jsonl ",
                     "events.jsonl.", "NUL.jsonl", "con.jsonl", "COM1.jsonl", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                with R.installed_io(event_log_name=name):
                    pass
        self.assertEqual((C.save, old._save_torch, old._save_array), self.original)
        self.assertFalse((self.out / "recovery").exists())


if __name__ == "__main__":
    unittest.main()
