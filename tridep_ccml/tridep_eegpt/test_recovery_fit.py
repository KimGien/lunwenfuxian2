"""Pure recovery contracts: no real model, forward, optimizer or cohort fit."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import recovery_fit as R


class AstContracts(unittest.TestCase):
    def test_only_seven_pending_identities_in_original_order(self):
        self.assertEqual(R.pending_identities(), [
            (R.T.CONDITIONS[1], 33, 2),
            *[(condition, 33, fold) for fold in (3, 4, 5) for condition in R.T.CONDITIONS]])
        self.assertEqual(len(R.completed_identities()), 23)
        self.assertEqual(len(set(R.identities())), 30)

    def test_complete_epoch_and_setup_ast_are_identical(self):
        original, generated = R.build_fit_ast()
        self.assertEqual(R._dump(R._epochs(original)), R._dump(R._epochs(generated)))
        self.assertEqual(R._dump(R._setup_nodes(original)), R._dump(R._setup_nodes(generated)))
        self.assertEqual(R.ast_evidence()["new_model_execution_during_proof"], 0)

    def test_original_source_ast_is_not_mutated(self):
        before = Path(R.T.__file__).read_bytes()
        first, _ = R.build_fit_ast()
        second, _ = R.build_fit_ast()
        self.assertEqual(R._dump(first), R._dump(second))
        self.assertEqual(Path(R.T.__file__).read_bytes(), before)

    def test_restoring_authorized_edits_recovers_exact_original_function(self):
        original, generated = R.build_fit_ast()
        restored = copy.deepcopy(generated)
        restored.name = original.name
        restored.body[0] = copy.deepcopy(original.body[0])
        restored.body.insert(3, copy.deepcopy(original.body[3]))
        restored.body[7] = copy.deepcopy(original.body[7])
        del restored.body[8]
        loops = [n for n in ast.walk(restored) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "condition"]
        inner = next(n for n in loops if isinstance(n.body[0], ast.Assign))
        del inner.body[1]
        changes = [i for i, node in enumerate(inner.body) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "meta"
                   and node.value.func.attr == "update"]
        self.assertEqual(len(changes), 1)
        del inner.body[changes[0]]
        self.assertEqual(R._dump(restored), R._dump(original))

    def test_unexpected_precheck_rejected(self):
        source = Path(R.T.__file__).read_text(encoding="utf-8")
        source = source.replace('for condition in CONDITIONS:\n        for seed in SEEDS:',
                                'for condition in CONDITIONS[::-1]:\n        for seed in SEEDS:', 1)
        with self.assertRaisesRegex(ValueError, "precheck"):
            R.build_fit_ast(source)

    def test_evidence_does_not_construct_model(self):
        with patch.object(R.T, "original_model", side_effect=AssertionError("No model")) as construct:
            proof = R.ast_evidence()
            self.assertTrue(proof["epoch_ast_sha256"]["identical"])
            self.assertTrue(proof["model_rng_dataset_loader_optimizer_ast_sha256"]["identical"])
            construct.assert_not_called()

    def test_entry_guard_precedes_any_features_or_model(self):
        _, generated = R.build_fit_ast()
        guard = types.SimpleNamespace(guard_training=lambda: (_ for _ in ()).throw(
            ValueError("Recovery action is not authorized")))
        namespace = dict(R.T.__dict__)
        namespace.update(_recovery=guard)
        exec(compile(ast.Module(body=[generated], type_ignores=[]), "synthetic_guard", "exec"), namespace)
        with patch.object(R.T, "original_model", side_effect=AssertionError("No model")) as construct:
            with patch.object(R.T, "load_features", side_effect=AssertionError("No features")) as load:
                with self.assertRaisesRegex(ValueError, "not authorized"):
                    namespace[generated.name]()
                construct.assert_not_called()
                load.assert_not_called()


class ReuseContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.values = [{"condition": c, "seed": s, "fold": f, "updates": 120,
                        "gradient_passes": 120, "epochs": 60, "training": R.T.TRAINING,
                        "initial_state_sha256": f"initial-{s}", "batch_order_sha256": f"order-{s}-{f}"}
                       for c, s, f in R.completed_identities()]
        self.recovery = types.SimpleNamespace(original_completed_metadata=lambda: self.values)
        self.out_patch = patch.object(R.T.C, "OUT", self.out)
        self.out_patch.start()

    def tearDown(self):
        self.out_patch.stop()
        self.temp.cleanup()

    def test_reuses_exact23_without_forward_or_model(self):
        with patch.object(R.T, "original_model", side_effect=AssertionError("No model")) as construct:
            completed = R._preload_completed(self.recovery)
            self.assertEqual(completed, self.values)
            self.assertIsNot(completed, self.values)
            construct.assert_not_called()

    def test_missing_complete_fit_rejected(self):
        self.values.pop()
        with self.assertRaisesRegex(ValueError, "exactly 23"):
            R._preload_completed(self.recovery)

    def test_reordered_complete_fit_rejected(self):
        self.values[0], self.values[1] = self.values[1], self.values[0]
        with self.assertRaisesRegex(ValueError, "identity/order"):
            R._preload_completed(self.recovery)

    def test_unbound_identity_rejected(self):
        self.values[0]["fold"] = 5
        with self.assertRaisesRegex(ValueError, "identity/order"):
            R._preload_completed(self.recovery)

    def test_any_pending_partial_folder_blocks_before_model(self):
        R.T._folder(*R.pending_identities()[0]).mkdir(parents=True)
        with patch.object(R.T, "original_model", side_effect=AssertionError("No model")) as construct:
            with self.assertRaisesRegex(ValueError, "explicit recovery review"):
                R._preload_completed(self.recovery)
            construct.assert_not_called()

    def test_reused_pair_enforces_same_initial_and_order(self):
        folder = self.out / "complete"
        folder.mkdir()
        meta = self.values[0]
        with patch.object(R.T.C, "read", return_value=meta):
            pair = R._reuse_completed_pair(meta, folder, None)
            self.assertEqual(pair, (meta["initial_state_sha256"], meta["batch_order_sha256"]))
            with self.assertRaisesRegex(ValueError, "paired"):
                R._reuse_completed_pair(meta, folder, ("wrong", pair[1]))

    def test_changed_reused_metadata_rejected(self):
        folder = self.out / "complete"
        folder.mkdir()
        with patch.object(R.T.C, "read", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "changed"):
                R._reuse_completed_pair(self.values[0], folder, None)


if __name__ == "__main__":
    unittest.main()
