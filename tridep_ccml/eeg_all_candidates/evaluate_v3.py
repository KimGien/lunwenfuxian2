"""Explicit evaluation of the unchanged FT math with qualified execution v3."""
from __future__ import annotations

import argparse
from pathlib import Path
import types

from . import evaluate_v2 as E2, protocol as P, storage as S
from . import execution_acceleration_v1 as A

SOURCE = Path(__file__).resolve()
ROOT = E2.ROOT
CONDITIONS = E2.CONDITIONS
selection_evidence = E2.selection_evidence


def execution_provenance(condition=None):
    value = E2.execution_provenance(condition)
    return {**value, "version": 3, "wrapper_source": S.desc(SOURCE),
            "delegated_evaluator_wrapper_source": S.desc(Path(E2.__file__)),
            "execution_acceleration_verifier_source": S.desc(Path(A.__file__)),
            "diagnostic_acceleration_does_not_change_test_window_chunk": True}


def verify_execution(value, condition=None):
    E2.E1.require(value == execution_provenance(condition), "Evaluator v3 identity differs")
    for record in value.values():
        if isinstance(record, dict) and set(record) == {"path", "sha256"}:
            S.checked(record)


def verify_training_execution(metadata):
    if "execution_acceleration" in metadata:
        return A.verify_execution_acceleration(metadata,
                                             condition=metadata["condition"], seed=metadata["seed"], fold=metadata["fold"])
    return None


def isolated_driver():
    wrapper = types.ModuleType(__package__ + "._evaluation_v3_wrapper")
    wrapper.__package__ = __package__
    wrapper.__file__ = E2.__file__
    exec(compile(Path(E2.__file__).read_text(encoding="utf-8"), E2.__file__, "exec"), wrapper.__dict__)
    wrapper.SOURCE = SOURCE
    wrapper.execution_provenance = execution_provenance
    wrapper.verify_execution = verify_execution
    original_materialize = wrapper.materialize_finetune

    def materialize(*args, **kwargs):
        metadata = kwargs.get("metadata", args[4] if len(args) > 4 else None)
        verify_training_execution(metadata)
        return original_materialize(*args, **kwargs)

    wrapper.materialize_finetune = materialize
    driver = wrapper.isolated_driver()
    original_context = driver._fit_context

    def context(condition, *args, **kwargs):
        metadata, plan, binding, destination = original_context(condition, *args, **kwargs)
        if P.CONDITION_BY_ID[condition].policy_key == "finetune":
            verified = verify_training_execution(metadata)
            if verified is not None:
                binding["training_execution_acceleration"] = verified
        return metadata, plan, binding, destination

    driver._fit_context = context
    return driver


def run(conditions=CONDITIONS, *, no_report=False):
    return isolated_driver().run(conditions, no_report=no_report)


def completed_condition_verified(condition):
    value = isolated_driver().completed_condition_verified(condition)
    value["evaluator_execution"] = execution_provenance(condition)
    return value


def completed_report_verified():
    value = isolated_driver().completed_report_verified()
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
