"""Read-only, synthetic loss-only parity diagnosis; no models or training.

Run from the workspace root. Optional --output writes a NEW JSON file only.
The author function and local expressions are extracted from the sealed sources.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def binding(path, lines):
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "lines": lines,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local_path = Path("tridep_ccml/fusion_eeg_grid_v2/models.py")
    author_path = Path("tridep_ccml/fusion_eeg_grid_v2/vendor/QMF/RGBD-scene-recognition/train_QMF_nyud2.py")
    engineering_path = Path("tridep_ccml/fusion_eeg_grid_v2/engineering.py")
    local_tree = ast.parse(local_path.read_text(encoding="utf-8"))
    author_tree = ast.parse(author_path.read_text(encoding="utf-8"))
    author_function = next(n for n in author_tree.body if isinstance(n, ast.FunctionDef) and n.name == "rank_loss")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[author_function], type_ignores=[]), str(author_path), "exec"), namespace)
    # Extract the actual nonzero-history target, margin and ranking expressions.
    local_assignments = {}
    for node in ast.walk(local_tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if (name, node.lineno) in {("target", 141), ("margin", 142), ("ranking", 144)}:
                local_assignments[name] = compile(ast.Expression(node.value), str(local_path), "eval")
    assert set(local_assignments) == {"target", "margin", "ranking"}

    history = np.asarray([0.8, 0.2, 0.0, 1.0], dtype=np.float64)

    class HistoryFixture:
        """Original normalized-history arithmetic, CPU output instead of CUDA."""
        def get_target_margin(self, index1, index2):
            score1 = (history[index1.numpy()] - history.min()) / (history.max() - history.min())
            score2 = (history[index2.numpy()] - history.min()) / (history.max() - history.min())
            difference = score1 - score2
            return torch.tensor(np.sign(difference), dtype=torch.float32), torch.tensor(abs(difference), dtype=torch.float32)

    examples = []
    for values in ([0.2, 0.4], [0.4, 0.2]):
        confidence = torch.tensor(values, requires_grad=True)
        indices = torch.tensor([0, 1])
        author_loss = namespace["rank_loss"](confidence, indices, HistoryFixture())
        author_loss.backward()
        author_gradient = confidence.grad.tolist()
        confidence.grad = None
        scores = history[:2]
        scope = {"torch": torch, "np": np, "x": torch.zeros(1), "diff": scores - np.roll(scores, -1),
                 "confidence": confidence, "ranking": torch.zeros(())}
        for name in ("target", "margin"):
            scope[name] = eval(local_assignments[name], scope)
        local_loss = eval(local_assignments["ranking"], scope)
        local_loss.backward()
        examples.append({
            "history_all": history.tolist(), "sample_indices": indices.tolist(),
            "normalized_sample_history": scores.tolist(), "confidence": values,
            "author_loss": author_loss.item(), "local_loss": local_loss.item(),
            "author_gradient": author_gradient, "local_gradient": confidence.grad.tolist(),
        })
    assert all(e["author_loss"] != e["local_loss"] for e in examples)
    report = {
        "schema": "qmf_source_parity_diagnosis_v1",
        "finding": "confirmed_undocumented_nonzero_history_ranking_difference",
        "local": binding(local_path, [136, 137, 139, 140, 141, 142, 143, 144]),
        "author": binding(author_path, list(range(57, 77))),
        "engineering": binding(engineering_path, list(range(52, 59))),
        "script": binding(Path(__file__), None),
        "author_commit": "fe6c4c6ef7cb23f0a89594ee413d485f1854268b",
        "definitions": "s = sign(normalized_history_i - normalized_history_j); m = abs(normalized_history_i - normalized_history_j); d = confidence_i - confidence_j",
        "author_loss_algebra": "max(0, s*d - m)",
        "local_loss_algebra": "max(0, s*d + m)",
        "examples": examples,
        "engineering_gap": "QMF author_parity checks energy-fusion output and a first-batch all-zero history finite/update-boundary case, not a nonzero-history ranking-value/gradient parity case.",
        "affected_scope": {"method": "QMF", "conditions": 16, "EST_fits": 225, "ST_fits": 15, "total_fits": 240,
                           "other_method_fits_not_implicated_by_this_finding": 960},
        "scientific_boundary": "The scores still describe actual trained local variants. This finding invalidates an unqualified exact-author-objective equivalence claim for QMF; it does not show that this difference caused low F1 or that a corrected/reconciled version would exceed baseline.",
        "paper": {"url": "https://proceedings.mlr.press/v202/zhang23ar/zhang23ar.pdf", "equations": [15, 16, 17],
                  "printed_eq16": "max(0, sign(w_i-w_j)*(kappa_i-kappa_j) + abs(w_i-w_j))",
                  "interpretation": "The printed equation also differs from the local formula's sign/margin variable placement. Do not assert that the local expression is strictly identical to the paper. Resolve and preregister the intended source before any new run."},
        "additional_primary_source_inspection": {
            "url": "https://github.com/QingyangZhang/QMF/blob/fe6c4c6ef7cb23f0a89594ee413d485f1854268b/text-image-classification/train_qmf.py",
            "lines": [180, 181, 182, 183, 184, 185, 186, 187, 188, 189],
            "finding": "This other author path also adds rank_margin/rank_target to the second confidence before using negative rank_target, matching the bound RGBD path's minus-margin algebra.",
        },
        "execution_scope": {"synthetic_scalar_losses_only": True, "model_instances": 0, "checkpoint_loads": 0,
                            "optimizer_updates": 0, "sealed_files_modified": False, "test_data_accessed": False},
        "torch_version": torch.__version__,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
