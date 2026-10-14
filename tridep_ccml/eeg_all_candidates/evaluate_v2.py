"""Versioned evaluation of truthful FTv2 checkpoints; original metrics stay fixed.

The immutable E1 implementation runs in isolated globals. Only the FT model
constructor, explicit execution provenance and selected/terminal update
disclosures are extended. No original module or source file is patched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import types

from . import evaluate as E1, protocol as P, registry as R, storage as S

SOURCE = Path(__file__).resolve()
ROOT = E1.ROOT
CONDITIONS = E1.CONDITIONS
REQUIRED_SELECTION_FIELDS = (
    "selected_checkpoint_phase", "selected_encoder_changed_parameter_count",
    "selected_head_changed_parameter_count", "terminal_encoder_changed_parameter_count",
    "terminal_head_changed_parameter_count",
)
ENCODER_STATE_FIELDS = (
    "selected_encoder_initial_state_sha256", "selected_encoder_state_sha256", "terminal_encoder_state_sha256",
)
CHANGE_NAME_FIELDS = (
    "selected_encoder_changed_parameter_names", "selected_head_changed_parameter_names",
    "terminal_encoder_changed_parameter_names", "terminal_head_changed_parameter_names",
)


def execution_provenance(condition=None):
    sources = {"wrapper_source": S.desc(SOURCE), "delegated_evaluator_source": S.desc(Path(E1.__file__)),
               "finetune_runner_source": S.desc(SOURCE.parent / "finetune_v2.py")}
    if condition is not None and P.CONDITION_BY_ID[condition].policy_key == "finetune":
        method = P.CONDITION_BY_ID[condition].method
        name = "luna_v2" if method == "LUNA" else ("codebrain_v2" if method == "CodeBrain" else method.lower())
        sources["actual_finetune_adapter_source"] = S.desc(SOURCE.parent / "adapters" / (name + ".py"))
    return {"version": 2, **sources, "technique": "isolated E1 globals with explicit FTv2 constructor and provenance overrides",
            "old_evaluator_and_global_modules_unchanged": True,
            "threshold_metrics_435_snapshots_retention_and_best_selection_unchanged": True}


def verify_execution(value, condition=None):
    E1.require(value == execution_provenance(condition), "Evaluator v2 wrapper/delegate/FT implementation identity differs")
    for item in value.values():
        if isinstance(item, dict) and set(item) == {"path", "sha256"}:
            S.checked(item)


def selection_evidence(metadata):
    """Disclose terminal training separately from the checkpoint actually scored."""
    fields = REQUIRED_SELECTION_FIELDS + ENCODER_STATE_FIELDS + CHANGE_NAME_FIELDS
    E1.require(all(name in metadata for name in fields), "FTv2 selected/terminal update evidence is missing")
    phase = metadata["selected_checkpoint_phase"]
    best = metadata["best_epoch"]
    expected = "head_only_warmup" if best <= 10 else "lastblock_finetune"
    E1.require(phase == expected, "Selected checkpoint phase does not match registered head-only warmup")
    for name in REQUIRED_SELECTION_FIELDS[1:]:
        E1.require(isinstance(metadata[name], int) and not isinstance(metadata[name], bool) and metadata[name] >= 0,
                   "Invalid actual parameter-change count: " + name)
    E1.require(metadata["terminal_encoder_changed_parameter_count"] > 0 and metadata["terminal_head_changed_parameter_count"] > 0,
               "Completed fine-tuning must show actual terminal encoder/head updates")
    if phase == "head_only_warmup":
        E1.require(metadata["selected_encoder_changed_parameter_count"] == 0,
                   "A head-only selected checkpoint cannot claim changed encoder parameters")
    for name in ENCODER_STATE_FIELDS:
        digest = metadata[name]
        E1.require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                   "Actual encoder-state fingerprint must be64 lowercase hex characters")
    for name in CHANGE_NAME_FIELDS:
        names = metadata[name]
        count_name = name.removesuffix("_names") + "_count"
        prefix = "encoder." if "encoder" in name else "head."
        E1.require(isinstance(names, list) and all(isinstance(n, str) and n.startswith(prefix) for n in names)
                   and len(set(names)) == len(names) == metadata[count_name], "Actual changed-parameter names/count differ")
    initial = metadata["selected_encoder_initial_state_sha256"]
    E1.require((metadata["selected_encoder_state_sha256"] != initial) == (metadata["selected_encoder_changed_parameter_count"] > 0)
               and metadata["terminal_encoder_state_sha256"] != initial, "Actual encoder fingerprints differ from changed-parameter evidence")
    return {"seed": metadata["seed"], "fold": metadata["fold"], "best_epoch": best,
            **{name: metadata[name] for name in fields},
            "terminal_update_proof_is_not_selected_encoder_update_proof": True}


class ExplicitStorage:
    """Delegate file operations; add source identity before the first write.

    This facade is installed only in the isolated evaluator namespace. The
    actual storage module and its global save function remain unchanged.
    """
    def __getattr__(self, name):
        return getattr(S, name)

    def save(self, path, value, *, immutable=False):
        target = Path(path).resolve()
        if target.is_relative_to(ROOT.resolve()) and target.name in ("complete.json", "evaluation_report.json", "predictions_manifest.json"):
            condition = value.get("condition")
            if condition is None and isinstance(value.get("binding"), dict):
                condition = value["binding"].get("condition")
            value["evaluator_execution"] = execution_provenance(condition)
        return S.save(path, value, immutable=immutable)


def materialize_finetune(condition, seed, fold, split, metadata, checkpoint, plan, snapshot):
    """The exact fixed best checkpoint is loaded through the actual FTv2 route."""
    import sys
    import torch
    from tridep_ccml.tridep_stage1 import common as C
    F2 = importlib.import_module(".finetune_v2", __package__)
    E1.require(P.CONDITION_BY_ID[condition].policy_key == "finetune" and snapshot == "best", "FTv2 evaluation requires fixed validation-selected best")
    if condition == "EEGMamba-lastblock-finetune":
        E1.require(sys.platform == "linux", "A complete qualified Linux Mamba prediction package is required")
        from .linux_entry import _require_runtime
        _require_runtime()
    C.runtime(seed)
    saved = torch.load(S.checked(checkpoint), map_location="cpu", weights_only=False)
    E1.require(isinstance(saved.get("state_dict"), dict) and all(bool(torch.isfinite(value).all()) for value in saved["state_dict"].values()),
               "Fixed FTv2 checkpoint is missing or nonfinite")
    E1.require(all(saved[name] == value for name, value in {
        "condition": condition, "seed": seed, "fold": fold, "registry": R.registration_descriptor(),
        "condition_source_lock": metadata["condition_source_lock"], "best_epoch": metadata["best_epoch"],
    }.items()), "FTv2 fixed checkpoint identity differs")
    selection_evidence(metadata)
    model, audit = F2.build_model(condition, seed=seed, device="cuda", window_chunk=metadata["window_chunk"])
    E1.require(audit["selected_encoder_parameter_names"] == metadata["adapter_audit"]["selected_encoder_parameter_names"],
               "Actual FTv2/LunaV2 trainable parameter contract differs from training")
    if condition == "EEGMamba-lastblock-finetune":
        for name in ("torch", "mamba_ssm", "causal_conv1d", "triton"):
            E1.require(audit["runtime"][name] == metadata["adapter_audit"]["runtime"][name], "Mamba runtime differs from actual FTv2 training")
    model.load_state_dict(saved["state_dict"], strict=True)
    E1.require(C.state_digest(model.state_dict()) == metadata["selected_state_sha256"], "FTv2 selected model fingerprint differs")
    model.requires_grad_(False).eval()
    return model


def isolated_driver():
    module = types.ModuleType(__package__ + "._evaluation_v2_runtime")
    module.__package__ = __package__
    module.__file__ = E1.__file__
    exec(compile(Path(E1.__file__).read_text(encoding="utf-8"), E1.__file__, "exec"), module.__dict__)
    module.SOURCE = SOURCE
    module.S = ExplicitStorage()
    original_context = module._fit_context
    original_materialize = module._materialize_predictor
    original_complete = module._verified_completed_fit
    original_summary = module.condition_summary
    original_comparison = module._comparison_contract
    module._delegated_fit_context = original_context
    module._delegated_materialize_predictor = original_materialize

    def context(condition, *args, **kwargs):
        metadata, plan, binding, destination = module._delegated_fit_context(condition, *args, **kwargs)
        binding["evaluator_execution"] = execution_provenance(condition)
        if P.CONDITION_BY_ID[condition].policy_key == "finetune":
            binding["selected_checkpoint_evidence"] = selection_evidence(metadata)
        return metadata, plan, binding, destination

    def materialize(condition, *args, **kwargs):
        if P.CONDITION_BY_ID[condition].policy_key == "finetune":
            return materialize_finetune(condition, *args, **kwargs)
        return module._delegated_materialize_predictor(condition, *args, **kwargs)

    def completed(path, binding, *args, **kwargs):
        result, rows = original_complete(path, binding, *args, **kwargs)
        verify_execution(result.get("evaluator_execution"), binding["condition"])
        return result, rows

    def summary(entries, cohort_ids):
        result = original_summary(entries, cohort_ids)
        if entries and all(e.get("condition") in CONDITIONS for e in entries):
            condition_ids = {e["condition"] for e in entries}
            E1.require(len(condition_ids) == 1, "Summarize one actual evaluation condition")
            condition = next(iter(condition_ids))
            if P.CONDITION_BY_ID[condition].policy_key == "finetune":
                evidence = [selection_evidence(S.read(S.checked(e["result"]["binding"]["metadata"]))) for e in entries]
                result["selected_checkpoint_evidence"] = sorted(evidence, key=lambda row: (row["seed"], row["fold"]))
        return result

    def comparison(condition, entries):
        result = original_comparison(condition, entries)
        if P.CONDITION_BY_ID[condition].policy_key == "finetune":
            selected = [e for e in entries if e["condition"] == condition and e["snapshot"] == "best"]
            evidence = [selection_evidence(S.read(S.checked(e["result"]["binding"]["metadata"]))) for e in selected]
            phases = Counter(row["selected_checkpoint_phase"] for row in evidence)
            result.update(selected_checkpoint_phases=json.dumps(dict(phases), sort_keys=True),
                          selected_encoder_changed_fits=sum(row["selected_encoder_changed_parameter_count"] > 0 for row in evidence),
                          terminal_encoder_changed_fits=sum(row["terminal_encoder_changed_parameter_count"] > 0 for row in evidence),
                          selected_update_evidence_source="evaluation_report conditions/variants/selected_checkpoint_evidence",
                          best_selection_and_retention_unchanged=True)
        else:
            result.update(selected_checkpoint_phases="not_applicable", selected_encoder_changed_fits="",
                          terminal_encoder_changed_fits="", selected_update_evidence_source="not_applicable",
                          best_selection_and_retention_unchanged=True)
        return result

    module._fit_context = context
    module._materialize_predictor = materialize
    module._verified_completed_fit = completed
    module.condition_summary = summary
    module._comparison_contract = comparison
    return module


def run(conditions=CONDITIONS, *, no_report=False):
    return isolated_driver().run(conditions, no_report=no_report)


def completed_condition_verified(condition):
    value = isolated_driver().completed_condition_verified(condition)
    value["evaluator_execution"] = execution_provenance(condition)
    return value


def completed_report_verified():
    driver = isolated_driver()
    value = driver.completed_report_verified()
    manifest = S.read(ROOT / "complete.json")
    verify_execution(manifest.get("evaluator_execution"))
    report = S.read(S.checked(manifest["evaluation_report"]))
    verify_execution(report.get("evaluator_execution"))
    value["evaluator_execution"] = execution_provenance()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    run(args.conditions, no_report=args.no_report)
