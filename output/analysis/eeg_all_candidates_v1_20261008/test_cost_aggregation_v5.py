"""Synthetic qualification resource inventory regressions, no GPU/tensors."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cost_v5_tests", ROOT / "cost_aggregation_v5.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class CostV5Tests(unittest.TestCase):
    def attempt(self, **changes):
        value = {"schema": "diagnostic_batching_qualification_attempt_v1", "kind": "engineering_qualification",
                 "status": "failed", "started_at": "2026-10-09T18:00:00+08:00", "seconds": 3.5,
                 "qualifier_source": C.X.descriptor(ROOT / "qualify_acceleration_20261009.py"),
                 "unflushed_crash_tail_exact": None, "gpu_stage_entered": True, "measurements": [],
                 "optimizer_updates": 0, "backward_calls": 0, "forward_started": 1, "forward_returned": 0}
        value.update(changes)
        return value

    def collect(self, values):
        v4 = C.load_v4()
        base = {"source": C.X.descriptor(C.DELEGATE), "cost_records": [], "category_observations": {}, "rules": [],
                "inputs": [], "input_read_errors": [], "max_observed_peak_allocated_bytes": None, "unknown_or_opaque_records": []}
        with tempfile.TemporaryDirectory(dir=C.WORK / "output", prefix="t_cost_") as temp:
            temp = Path(temp).resolve()
            self.assertTrue(temp.is_relative_to((C.WORK / "output").resolve()))
            for i, value in enumerate(values):
                path = temp / "execution_acceleration_v1/C/qualification_attempts" / str(i) / "attempt.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(value), encoding="utf8")
            with patch.object(v4, "OUT", temp), patch.object(v4, "collect", return_value=deepcopy(base)), patch.object(C, "load_v4", return_value=v4):
                return C.collect()

    def test_failure_peak_before_first_measurement_is_retained(self):
        result = self.collect([self.attempt(peak_allocated_bytes_observed_at_failure=128)])
        row, = result["cost_records"]
        self.assertEqual(128, row["peak_allocated_bytes"])
        self.assertEqual(["peak_allocated_bytes_observed_at_failure"], row["peak_source_fields"])
        self.assertEqual(128, result["max_observed_peak_allocated_bytes"])

    def test_failure_peak_and_completed_measurement_use_max_not_sum(self):
        result = self.collect([self.attempt(peak_allocated_bytes_observed_at_failure=128, measurements=[{"peak_allocated_bytes": 96, "synchronized_wall_seconds": 100}])])
        row, = result["cost_records"]
        self.assertEqual(128, row["peak_allocated_bytes"])
        self.assertEqual(3.5, result["category_observations"]["diagnostic_batching_qualification"]["known_recorded_scope_seconds_subtotal"])
        self.assertFalse(row["nested_candidate_measurement_seconds_added"])

    def test_unknown_peak_and_time_stay_null(self):
        attempt = self.attempt(status="running", peak_allocated_bytes_observed_at_failure=None)
        del attempt["seconds"]
        result = self.collect([attempt])
        self.assertIsNone(result["cost_records"][0]["peak_allocated_bytes"])
        self.assertIsNone(result["category_observations"]["diagnostic_batching_qualification"]["known_recorded_scope_seconds_subtotal"])

    def test_true_recorded_zero_seconds_preserved(self):
        result = self.collect([self.attempt(seconds=0)])
        self.assertEqual(0, result["category_observations"]["diagnostic_batching_qualification"]["known_recorded_scope_seconds_subtotal"])

    def test_boolean_failure_peak_is_rejected_not_resource(self):
        result = self.collect([self.attempt(peak_allocated_bytes_observed_at_failure=True)])
        self.assertEqual([], result["cost_records"])
        self.assertEqual(1, len(result["input_read_errors"]))

    def test_exact_duplicate_attempt_not_double_counted(self):
        value = self.attempt()
        result = self.collect([value, value])
        category = result["category_observations"]["diagnostic_batching_qualification"]
        self.assertEqual((1, 1, 3.5), (category["primary_invocations"], category["supplemental_records"], category["known_recorded_scope_seconds_subtotal"]))
        self.assertEqual(2, len(result["inputs"]))


if __name__ == "__main__": unittest.main(verbosity=2)
