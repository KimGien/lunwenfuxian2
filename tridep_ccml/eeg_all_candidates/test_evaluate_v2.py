"""Synthetic standard-library checks: no models, signals, GPU or real tests."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from . import evaluate as E1, evaluate_v2 as E2


def selection(*, best=8, selected_encoder=0):
    return {"seed": 11, "fold": 1, "best_epoch": best,
            "selected_checkpoint_phase": "head_only_warmup" if best <= 10 else "lastblock_finetune",
            "selected_encoder_changed_parameter_count": selected_encoder, "selected_head_changed_parameter_count": 8,
            "terminal_encoder_changed_parameter_count": 17, "terminal_head_changed_parameter_count": 8,
            "selected_encoder_initial_state_sha256": "a" * 64,
            "selected_encoder_state_sha256": ("b" if selected_encoder else "a") * 64,
            "terminal_encoder_state_sha256": "c" * 64,
            "selected_encoder_changed_parameter_names": ["encoder.synthetic_" + str(i) for i in range(selected_encoder)],
            "selected_head_changed_parameter_names": ["head.synthetic_" + str(i) for i in range(8)],
            "terminal_encoder_changed_parameter_names": ["encoder.synthetic_" + str(i) for i in range(17)],
            "terminal_head_changed_parameter_names": ["head.synthetic_" + str(i) for i in range(8)]}


class EvaluatorV2CPU(unittest.TestCase):
    def test_all_public_routes_gate_before_data_model_or_writes(self):
        for function, args in ((E2.run, ()), (E2.completed_condition_verified, ("LUNA-lastblock-finetune",)),
                               (E2.completed_report_verified, ())):
            with patch.object(E1.R, "assert_test_unlocked", side_effect=ValueError("360 incomplete")) as gate, \
                 patch.object(E2, "materialize_finetune") as model, patch.object(E2.S, "read") as read, \
                 patch.object(E2.S, "save") as save:
                with self.assertRaisesRegex(ValueError, "360 incomplete"):
                    function(*args)
                gate.assert_called_once_with()
                model.assert_not_called()
                read.assert_not_called()
                save.assert_not_called()

    def test_isolation_leaves_original_source_storage_and_algorithms_unchanged(self):
        source, save, materialize = E1.SOURCE, E1.S.save, E1._materialize_predictor
        driver = E2.isolated_driver()
        self.assertEqual(driver.SOURCE, E2.SOURCE)
        self.assertEqual(E1.SOURCE, source)
        self.assertIs(E1.S.save, save)
        self.assertIs(E1._materialize_predictor, materialize)
        self.assertEqual(driver.metrics.__code__.co_code, E1.metrics.__code__.co_code)
        self.assertEqual(driver.retention.__code__.co_code, E1.retention.__code__.co_code)
        self.assertEqual(driver.matched_control.__code__.co_code, E1.matched_control.__code__.co_code)
        self.assertEqual(driver.CONDITIONS, E1.CONDITIONS)

    def test_luna17_ft_routes_v2_while_non_ft_delegates_original_constructor(self):
        driver = E2.isolated_driver()
        with patch.object(E2, "materialize_finetune", return_value="synthetic_ftv2") as ft, \
             patch.object(driver, "_delegated_materialize_predictor", return_value="synthetic_original") as delegated:
            self.assertEqual(driver._materialize_predictor("LUNA-lastblock-finetune", "synthetic_17_parameter_contract"), "synthetic_ftv2")
            ft.assert_called_once_with("LUNA-lastblock-finetune", "synthetic_17_parameter_contract")
            delegated.assert_not_called()
            self.assertEqual(driver._materialize_predictor("LUNA-frozen-smallhead", "synthetic_original_head"), "synthetic_original")
            delegated.assert_called_once_with("LUNA-frozen-smallhead", "synthetic_original_head")

    def test_true_source_and_selected_phase_are_added_before_probability_binding(self):
        driver = E2.isolated_driver()
        old = {"evaluator_source": {"path": "synthetic", "sha256": "a" * 64}}
        evidence = selection()
        with patch.object(driver, "_delegated_fit_context", return_value=(evidence, {}, old, Path("synthetic_not_written"))):
            _, _, binding, _ = driver._fit_context("LUNA-lastblock-finetune")
        execution = binding["evaluator_execution"]
        self.assertEqual(execution["wrapper_source"], E2.S.desc(E2.SOURCE))
        self.assertEqual(execution["delegated_evaluator_source"], E2.S.desc(E1.__file__))
        self.assertTrue(execution["actual_finetune_adapter_source"]["path"].endswith("/luna_v2.py"))
        self.assertEqual(binding["selected_checkpoint_evidence"]["selected_encoder_changed_parameter_count"], 0)
        E2.verify_execution(execution, "LUNA-lastblock-finetune")
        wrong = copy.deepcopy(execution)
        wrong["wrapper_source"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "implementation identity"):
            E2.verify_execution(wrong, "LUNA-lastblock-finetune")

    def test_source_facade_augments_first_immutable_complete_without_global_patch(self):
        facade = E2.ExplicitStorage()
        complete = {"passed": True, "binding": {"condition": "LUNA-lastblock-finetune"}}
        with patch.object(E2.S, "save") as original:
            facade.save(E2.ROOT / "synthetic_only/complete.json", complete, immutable=True)
            self.assertTrue(original.call_args.kwargs["immutable"])
            self.assertIn("evaluator_execution", original.call_args.args[1])
            self.assertEqual(original.call_args.args[1]["evaluator_execution"]["delegated_evaluator_source"], E2.S.desc(E1.__file__))
        self.assertNotIsInstance(E1.S, E2.ExplicitStorage)

    def test_warmup_best_is_legal_but_cannot_claim_selected_encoder_change(self):
        value = E2.selection_evidence(selection(best=8))
        self.assertEqual(value["selected_checkpoint_phase"], "head_only_warmup")
        self.assertEqual(value["selected_encoder_changed_parameter_count"], 0)
        self.assertEqual(value["terminal_encoder_changed_parameter_count"], 17)
        with self.assertRaisesRegex(ValueError, "head-only selected checkpoint"):
            E2.selection_evidence(selection(best=8, selected_encoder=17))
        late = E2.selection_evidence(selection(best=25, selected_encoder=17))
        self.assertEqual(late["selected_checkpoint_phase"], "lastblock_finetune")
        self.assertEqual(late["terminal_encoder_state_sha256"], "c" * 64)
        tampered = selection(best=25, selected_encoder=17)
        tampered["selected_encoder_changed_parameter_names"].pop()
        with self.assertRaisesRegex(ValueError, "names/count"):
            E2.selection_evidence(tampered)
        candidate = {"primary": {"macro_f1": .42, "balanced_accuracy": .5, "mdd_recall": .66, "auc": .6, "nll": .6, "brier": .2},
                     "seeds": {str(seed): {"fold_mean": {"macro_f1": .42}} for seed in E1.P.SEEDS}}
        control = {"primary": {"macro_f1": .4, "balanced_accuracy": .5, "mdd_recall": .7, "auc": .6, "nll": .7, "brier": .25},
                   "seeds": {str(seed): {"fold_mean": {"macro_f1": .4}} for seed in E1.P.SEEDS}}
        self.assertEqual(E2.isolated_driver().retention(candidate, control), E1.retention(candidate, control))


if __name__ == "__main__":
    unittest.main(verbosity=2)
