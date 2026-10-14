"""Isolated stdlib IO measurements; no model, CUDA, raw signal or fit work."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S


def main():
    assert S.desc(Path(S.__file__))["sha256"] == "1c5dc51e33633beee8b6b54ffd43249ac53679486f62db18e9872e67fbaf0fbe"
    sample = S.OUT / "finetune_v2_attempts/CSBrain-lastblock-finetune/e_1791520704920988100_1e347180/attempt.json"
    payload = S.read(sample)
    assert payload["status"] == "complete" and payload["kind"] == "engineering"
    token = str(time.time_ns())
    folder = S.REPORT / "observer_io_benchmarks" / token
    path = folder / "benchmark_copy.json"
    timings = []
    for index in range(64):
        value = dict(payload, benchmark_copy_only=True, benchmark_index=index)
        start = time.perf_counter()
        S.save(path, value)
        timings.append(time.perf_counter() - start)
    sorted_times = sorted(timings)
    result = {"source": S.desc(Path(__file__)), "storage_source": S.desc(Path(S.__file__)),
        "input_payload": S.desc(sample), "captured_at": S.now(),
        "kind": "isolated_cpu_durable_json_write_benchmark", "formal_fits": 0,
        "model_constructions": 0, "gpu_calls": 0, "optimizer_updates": 0,
        "raw_or_feature_arrays_read": False, "test_probabilities_read": False,
        "writes": len(timings), "payload_bytes": len(json.dumps(payload,ensure_ascii=False,indent=2).encode()),
        "write_seconds": timings, "total_seconds": sum(timings),
        "mean_seconds": statistics.mean(timings), "median_seconds": statistics.median(timings),
        "p95_seconds": sorted_times[int(.95*(len(timings)-1))],
        "benchmark_last_file": S.desc(path),
        "scope": "Same durable storage.save on a separate report-directory copy while training runs. Not an attribution of actual training wall time.",
        "inferred_710_trace_writes_seconds_at_sample_mean": statistics.mean(timings)*710,
        "actual_training_io_fraction": None}
    report = S.REPORT / ("observer_io_benchmark_" + token + ".json")
    S.save(report, result, immutable=True)
    print(json.dumps({"artifact": S.desc(report), "writes":result["writes"],
        "mean_ms":result["mean_seconds"]*1000,"median_ms":result["median_seconds"]*1000,
        "p95_ms":result["p95_seconds"]*1000,"sample_710_writes_seconds":result["inferred_710_trace_writes_seconds_at_sample_mean"]}))


if __name__ == "__main__":
    main()
