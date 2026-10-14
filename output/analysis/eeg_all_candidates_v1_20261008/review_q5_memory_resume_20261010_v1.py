"""Independent immutable JSON/source review; never queries or controls processes."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "output/analysis/eeg_all_candidates_v1_20261008"
OUT = ROOT / "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008"
PAUSE = OUT / "pause_events/q5_user_pause_1791607365360"
RESUME = OUT / "pause_events/q5_memory_resume_1791617874108"
SOURCE = REPORT / "resume_paused_q5_20261010_v1.ps1"
SOURCE_SHA = "1f227764bc5b82351067319d0815e5d5926bb80baf3a7446b2ff55f761684761"
PAUSE_SHA = "fcbd717850d8cee8bc57ceb0594cf8dab482d10363f618f8d5cae7406bc37b3b"
NATIVE_SHA = "acb9b67e4a15548cbcab476fb308ce29bfdc8e2d61a2fd35a39337cce5ac42b2"
OWNER_SHA = "82cac0a01c3ecafe17dbfa7501ac2118873c54f912931bb742dc8ad853ba0d65"
EXPECTED = [
    ("gpu_child", 31128, 16096, "134360741986202743", 64),
    ("launcher", 16096, 45980, "134360741986106595", 1),
    ("dispatcher", 45980, 38860, "134360707154222972", 1),
    ("dispatcher_launcher", 38860, 50164, "134360707153787861", 1),
]


def require(value, message):
    if not value:
        raise ValueError(message)


class Reader:
    def __init__(self):
        self.inputs = {}

    def path(self, path):
        p = Path(path)
        p = (p if p.is_absolute() else ROOT / p).resolve()
        require(p.is_relative_to(ROOT), "Input outside workspace")
        require(p.suffix.lower() in {".json", ".py", ".ps1"}, "Non-JSON/source read forbidden")
        return p

    def data(self, path, expected=None):
        p = self.path(path)
        data = p.read_bytes()
        desc = {"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}
        require(expected is None or desc["sha256"] == expected, "Input SHA mismatch: " + desc["path"])
        self.inputs[desc["path"]] = desc
        return data

    def load(self, path, expected=None):
        return json.loads(self.data(path, expected).decode("utf-8-sig"))

    def descriptor(self, path, expected=None):
        self.data(path, expected)
        return self.inputs[self.path(path).relative_to(ROOT).as_posix()]


def rows(value):
    # PowerShell serializes a one-element Threads-Of result as an object.
    require(isinstance(value, (list, dict)), "Invalid row container")
    return value if isinstance(value, list) else [value]


def keyed(value, key="role"):
    value = rows(value)
    result = {r[key]: r for r in value}
    require(len(result) == len(value), "Duplicate evidence identity")
    return result


def qpc(clock):
    require(clock["qpc_frequency"] == "10000000", "Unexpected QPC frequency")
    require(isinstance(clock["wall"], str) and clock["wall"], "Missing wall timestamp")
    return int(clock["qpc"])


def main():
    r = Reader()
    source = r.descriptor(SOURCE, SOURCE_SHA)
    pause = r.load(PAUSE / "pause_complete.json", PAUSE_SHA)
    old = r.load(PAUSE / "native_pause_complete.json", NATIVE_SHA)
    for descriptor in (pause["source"], pause["native_control_source"], old["source"]):
        r.descriptor(descriptor["path"], descriptor["sha256"])
    pre = r.load(RESUME / "captured_resume_preflight.json")
    done = r.load(RESUME / "native_resume_complete.json")
    released = r.load(RESUME / "resume_controls_released.json")
    intent = r.load(PAUSE / "resume_intent.json")
    for item in (pre, done, released, intent):
        require(item["source"] == source, "Resume source not exactly bound")
    require(r.path(pre["pause_proof"]) == PAUSE / "pause_complete.json" and pre["pause_proof_sha256"] == PAUSE_SHA, "Pause proof binding")
    require(r.path(pre["native_pause"]) == PAUSE / "native_pause_complete.json" and pre["native_pause_sha256"] == NATIVE_SHA, "Native proof binding")
    require(Path(intent["folder"]).resolve() == RESUME, "Intent folder mismatch")
    require(intent["explicit_user_continue"] is True and intent["native_retry_permitted"] is False, "Explicit single-use authority")
    require(pause["status"] == "paused_by_user" and old["status"] == "paused_by_explicit_user", "Original pause incomplete")
    require(done["status"] == "same_four_objects_resumed_by_explicit_continue", "Resume incomplete")
    require(released["status"] == "explicit_continue_controls_archived_then_released", "Controls release incomplete")
    require(pre["owner_sha256"] == OWNER_SHA, "Owner SHA mismatch")
    owner = r.load(PAUSE / "q5_owner.json", OWNER_SHA)
    require(owner == pre["owner_snapshot"] == pause["owner_snapshot"] == old["owner_snapshot"], "Immutable owner mismatch")
    require(owner["id"] == "069c7431c2c242c095474486f70e276f" and owner["pid"] == 45980 and owner["child_pid"] == 16096 and owner["child_job"] == "REVE-lastblock-finetune-fit", "Owner identity mismatch")
    old_ids, old_threads = keyed(old["identities"]), keyed(old["verification"])
    pre_ids, post_ids = keyed(pre["identities"]), keyed(done["verification"])
    suspend_actions, resume_actions = keyed(old["actions"]), keyed(done["actions"])
    expected_roles = {e[0] for e in EXPECTED}
    for value in (old_ids, old_threads, pre_ids, post_ids, suspend_actions, resume_actions):
        require(set(value) == expected_roles, "Exactly four role identities required")
    require([x["role"] for x in done["actions"]] == [e[0] for e in EXPECTED], "Native resume order differs")
    event_names = set()
    proof_rows = []
    last_qpc = qpc(pre["clock"])
    require(qpc(intent["created"]) <= last_qpc, "Intent must precede native calls")
    for seq, (role, pid, parent, birth, count) in enumerate(EXPECTED, 1):
        before, current, after = old_ids[role], pre_ids[role], post_ids[role]
        require(before["pid"] == current["pid"] == after["pid"] == pid, "PID mismatch")
        require(before["creation_100ns"] == current["creation_100ns"] == after["creation_100ns"] == birth, "Native birth mismatch")
        require(before["parent_pid"] == current["parent"] == parent, "Parent mismatch")
        require(before["image"] == current["image"] and before["command_line"] == current["command_line"], "Image/command mismatch")
        original = keyed(old_threads[role]["threads"], "id")
        captured = keyed(current["threads"], "id")
        observed = keyed(after["threads"], "id")
        require(len(original) == count and set(original) == set(captured) == set(observed), "Original thread identities differ")
        require(all(t["state"] == "Wait" and t["wait_reason"] == "Suspended" for t in list(original.values()) + list(captured.values())), "Original thread not suspended before resume")
        require(all(t["wait_reason"] != "Suspended" for t in observed.values()), "Thread remained suspended")
        require(current["added_threads"] == [], "Unexpected added threads at actual resume")
        started_name = f"event_{seq:02}_{role}_started.json"
        returned_name = f"event_{seq:02}_{role}_returned.json"
        event_names.update((started_name, returned_name))
        started = r.load(RESUME / started_name)
        returned = r.load(RESUME / returned_name)
        require(started["source"] == source and started["role"] == role and started["pid"] == pid and started["birth"] == birth, "Started event identity mismatch")
        require(started["resume_started"] is True and started["native_result"] is None, "Started event must precede result")
        require(returned == resume_actions[role], "Returned event differs from terminal action")
        require(returned["pid"] == pid and returned["creation_100ns"] == birth and type(returned["native_status"]) is int and returned["native_status"] == 0, "Resume native failure")
        require(suspend_actions[role]["pid"] == pid and type(suspend_actions[role]["native_status"]) is int and suspend_actions[role]["native_status"] == 0, "Original native suspension failure")
        require(last_qpc <= qpc(started["clock"]) <= qpc(returned["returned"]), "Started/returned QPC ordering")
        last_qpc = qpc(returned["returned"])
        proof_rows.append({"role": role, "pid": pid, "parent_pid": parent, "creation_100ns": birth, "image": current["image"], "command_line": current["command_line"], "original_threads": count, "original_thread_ids": sorted(original), "pre_resume_all_original_suspended": True, "post_resume_original_suspended": 0, "added_threads_at_actual_resume": [], "native_suspend_calls_in_bound_pause": 1, "native_resume_calls_in_bound_resume": 1, "native_resume_status": 0, "resume_returned_clock": returned["returned"], "post_resume_states": sorted({(t["state"], t["wait_reason"] or "") for t in observed.values()})})
    require({p.name for p in RESUME.glob("event_*.json")} == event_names, "Extra/missing native event files")
    require(all(last_qpc <= qpc(x["clock"]) <= qpc(done["clock"]) for x in post_ids.values()), "Post-resume verification ordering")
    require(qpc(done["clock"]) <= qpc(released["clock"]), "Release before verification")
    controls = pre["controls"]
    require(controls == done["controls_archived"] == released["controls"] and len(controls) == 4, "Archive/release linkage")
    expected_controls = [OUT / "user_pause_q5_owner.json"] + [OUT / "execution_acceleration_v1" / n / "user_pause.json" for n in ("sync_boundary", "sync_boundary_v2", "sync_boundary_v3")]
    for index, control in enumerate(controls):
        require(Path(control["original"]).resolve() == expected_controls[index], "Unexpected released control")
        require(r.path(control["archive"]) == RESUME / f"control_{index+1}.json", "Archive location mismatch")
        value = r.load(control["archive"], control["sha256"])
        if index:
            require(value == {"paused": True} and type(value["paused"]) is bool, "Archived strict marker invalid")
        else:
            require(Path(value["folder"]).resolve() == PAUSE and value["automatic_resume_authorized"] is False, "Archived pause owner invalid")
    checked_snapshots, skipped_binary_descriptors = [], []
    for entry in pause["retained"]:
        require(entry["original"]["sha256"] == entry["snapshot"]["sha256"], "Snapshot descriptor mismatch")
        desc = entry["snapshot"]
        if Path(desc["path"]).suffix == ".json":
            require(r.path(desc["path"]).parent == PAUSE, "Snapshot must be immutable pause copy")
            r.load(desc["path"], desc["sha256"])
            checked_snapshots.append(desc)
        else:
            require(Path(desc["path"]).suffix == ".pt", "Unexpected unverified snapshot type")
            skipped_binary_descriptors.append(desc)
    for field in ("new_queue_processes", "optimizer_operations_by_controller", "model_reads_by_controller", "process_terminate_calls"):
        require(type(done[field]) is int and done[field] == 0, "Unexpected controller operation")
    require(type(done["native_resume_calls"]) is int and done["native_resume_calls"] == 4, "Native call total")
    require(done["exact_active_elapsed_seconds"] is None and pre["paused_interval_qpc_exact"] is None and done["pause_boundary_qpc_not_measured"] is True and pre["pause_boundary_qpc_not_measured"] is True, "Unmeasured active duration must remain null")
    require(done["pause_not_training"] is True and pause["pause_interval_not_training"] is True, "Pause accounting must be separate")
    report = {
        "schema": "independent_q5_memory_resume_review_v1", "passed": True,
        "review_source": r.descriptor(Path(__file__)), "resume_source": source,
        "resume_folder": RESUME.relative_to(ROOT).as_posix(), "processes": proof_rows,
        "same_four_original_objects": True, "total_original_threads_verified": 67,
        "total_original_threads_still_suspended_after_recorded_resume": 0,
        "new_EventPairLow_thread_note": "Previously observed extra thread was absent in actual immutable resume preflight; added_threads is empty for all four roles. No extra native call was made for it.",
        "controls_archive_hashes_verified": controls,
        "controls_release_evidence": r.descriptor(RESUME / "resume_controls_released.json"),
        "controls_current_paths_not_read_or_tested": True,
        "immutable_json_snapshots_verified": checked_snapshots,
        "checkpoint_descriptors_carried_without_open_or_rehash": skipped_binary_descriptors,
        "pause_clock": {"paused_at": pause["paused_at"], "pause_boundary_qpc": None},
        "resume_preflight_clock": pre["clock"], "resume_complete_clock": done["clock"],
        "exact_pause_elapsed_seconds": None, "exact_active_elapsed_seconds": None,
        "timing_limitation": "Resume-side QPC ordering is verified; the pause boundary had no QPC. Wall timestamps do not establish exact active training time. Same-process elapsed timers may span suspension; preserve original timers and disclose the pause separately without inventing an exact subtraction.",
        "resume_evidence_scope": "Immutable records prove one status-zero native resume per original process and original-thread release at that observation. No live process scan or current training-progress read was performed. This is a resume review, not a fit-completion or whole-plan certificate.",
        "review_io": {"torch_imports": 0, "tensor_loads": 0, "binary_checkpoint_or_weight_reads": 0, "model_constructions": 0, "GPU_calls": 0, "process_queries": 0, "process_controls": 0, "active_progress_reads": 0, "raw_or_features_or_test_probability_reads": 0},
        "inputs": list(r.inputs.values()),
    }
    target = REPORT / f"independent_q5_memory_resume_review_{time.time_ns()}.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"report": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "passed": True, "original_threads": 67}, ensure_ascii=False))


if __name__ == "__main__":
    main()
