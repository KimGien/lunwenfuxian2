"""Preregistered EEG-only protocol; importing this module performs no fitting.

The immutable registration has 23 supervised conditions and one fold-local
self-supervised condition, each with three seeds and the unchanged five folds.
Author-version input contracts and exact trainable parameter names must be
locked separately before a condition may enter the GPU queue.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from statistics import mean
from typing import Sequence

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008"
REPORT = WORK / "output/analysis/eeg_all_candidates_v1_20261008"
PLAN = WORK / "output/plans/EEG单模态九篇候选全覆盖与长训练改进计划_v1_20261008.md"
EXPERIMENT_ID = OUT.name
SEEDS = (11, 22, 33)
FOLDS = (1, 2, 3, 4, 5)
FOLDS_SHA256 = "a28ebe92c94428128313586d3a5524b2f03befcbc473215f4b0b2dcb49c3284d"
METHODS = ("CSBrain", "REVE", "EEGPT", "LUNA", "CodeBrain", "EEGConformer", "BIOT", "EEG2Rep", "EEGMamba")
PRETRAINED_METHODS = ("EEGPT", "CSBrain", "REVE", "LUNA", "CodeBrain", "BIOT", "EEGMamba")
CHANNELS = ("FP2", "FP1", "F7", "F3", "F4", "F8", "FCz", "C3", "C4", "T3", "CPz", "T4", "P3", "Pz", "P4", "T5", "T6", "O1", "O2")
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BIOT_BIPOLAR = ("FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2")
SNAPSHOT_EPOCHS = (60, 120, 200, 300)
PRIMARY_DIAGNOSTIC_COMPARISON = (60, 300)


@dataclass(frozen=True)
class TrainingPolicy:
    initial_epochs: int
    minimum_stop_epoch: int
    patience: int
    final_epochs: int
    optimizer: str = "AdamW"
    head_lr: float = 1e-3
    backbone_lr: float | None = None
    weight_decay: float = 1e-3
    warmup_epochs: int = 10
    backbone_warmup_epochs: int = 0
    gradient_clip: float | None = 1.0
    minimum_improvement: float = 1e-4
    early_stopping: bool = True
    checkpoint_selection: str = "validation_subject_mean_ce"
    effective_subject_batch: int | None = 8
    logical_window_batch: int | None = None
    scheduler: str = "warmup_cosine_to_0.1_initial_then_hold_on_extension"


POLICIES = {
    "frozen": TrainingPolicy(300, 60, 60, 500),
    "finetune": TrainingPolicy(120, 40, 30, 200, head_lr=3e-4, backbone_lr=1e-5,
                               weight_decay=1e-2, backbone_warmup_epochs=5),
    "scratch": TrainingPolicy(500, 100, 80, 800, head_lr=3e-4, backbone_lr=3e-4),
    "ssl": TrainingPolicy(300, 60, 60, 500, head_lr=1e-4, backbone_lr=1e-4,
                          weight_decay=1e-2, checkpoint_selection="fixed_validation_mask_author_ssl_loss",
                          effective_subject_batch=None, logical_window_batch=64),
    "ssl_supervised": TrainingPolicy(200, 60, 50, 300, head_lr=3e-4, backbone_lr=1e-4),
    "diagnostic": TrainingPolicy(300, 300, 0, 300, optimizer="Adamax", head_lr=.005,
                                 weight_decay=0., warmup_epochs=0, gradient_clip=None,
                                 early_stopping=False, checkpoint_selection="fixed_epoch_300",
                                 effective_subject_batch=16, scheduler="none"),
}


@dataclass(frozen=True)
class Condition:
    condition_id: str
    method: str
    stage: str
    objective: str
    policy_key: str
    input_seconds: int
    route: str
    control: bool = False
    dependency: str | None = None


DIAGNOSTIC_CONDITIONS = (
    "EEGPT-mean-projection4s-originalCNN-300ep",
    "LaBraM-native4s-originalCNN-300ep",
)


def _conditions() -> tuple[Condition, ...]:
    entries = [Condition(DIAGNOSTIC_CONDITIONS[0], "EEGPT", "1", "supervised", "diagnostic", 4, "legacy_projection_original_cnn"),
               Condition(DIAGNOSTIC_CONDITIONS[1], "LaBraM", "1", "supervised", "diagnostic", 4, "legacy_original_cnn", True)]
    entries.extend(Condition(f"{method}-frozen-smallhead", method, "2", "supervised", "frozen", 4 if method == "EEGPT" else 5, "native_frozen_smallhead") for method in PRETRAINED_METHODS)
    entries.extend((Condition("EEGConformer-scratch", "EEGConformer", "2", "supervised", "scratch", 5, "author_backbone_from_scratch"),
                    Condition("EEG2Rep-supervised", "EEG2Rep", "2", "supervised", "ssl_supervised", 5, "fold_local_ssl_then_supervised", dependency="EEG2Rep-fold-local-ssl"),
                    Condition("CBraMod-frozen-smallhead", "CBraMod", "2", "supervised", "frozen", 5, "native_frozen_smallhead", True),
                    Condition("LaBraM-native5s-frozen-smallhead", "LaBraM", "2", "supervised", "frozen", 5, "native_frozen_smallhead", True),
                    Condition("LaBraM-native4s-frozen-smallhead", "LaBraM", "2", "supervised", "frozen", 4, "native_frozen_smallhead", True),
                    Condition("EEG2Rep-fold-local-ssl", "EEG2Rep", "2附属", "self_supervised", "ssl", 5, "author_ssp_ema_vicreg_fold_local")))
    entries.extend(Condition(f"{method}-lastblock-finetune", method, "3", "supervised", "finetune", 4 if method == "EEGPT" else 5, "official_pretrained_lastblock_and_fresh_head") for method in PRETRAINED_METHODS)
    entries.extend(Condition(f"LaBraM-native{seconds}s-lastblock-finetune", "LaBraM", "3", "supervised", "finetune", seconds, "official_pretrained_lastblock_and_fresh_head", True) for seconds in (5, 4))
    return tuple(entries)


CONDITIONS = _conditions()
CONDITION_BY_ID = {condition.condition_id: condition for condition in CONDITIONS}
EXPECTED_SUPERVISED_FITS = 345
EXPECTED_SSL_FITS = 15
EXPECTED_FORMAL_FITS = 360
UNFREEZE_RULES = {
    "EEGPT": "last encoder block plus fresh small head",
    "CSBrain": "last spatiotemporal encoding block and output normalization",
    "REVE": "last encoder block and output normalization",
    "LUNA": "last temporal encoder block, pooling queries and output normalization",
    "CodeBrain": "last EEGSSM representation block and output normalization",
    "BIOT": "last Transformer block and output normalization",
    "EEGMamba": "last bidirectional Mamba block and output normalization",
    "LaBraM": "last encoder block and output normalization",
}
DATA_RULES = {
    "cohort_id": "MODMA-37-v1", "subjects": 37, "mdd": 16, "hc": 21,
    "sample_rate_hz": 200, "windows_per_subject": 60, "samples_per_window": 1000,
    "raw_shape_per_subject": [60, 19, 1000], "channels": list(CHANNELS),
    "seed_changes_split": False, "train_count": [26, 27], "val_count": 3, "test_count": [7, 8],
    "all_fitted_preprocessing_train_people_only": True,
    "frozen_external_encoder_shared_cache_allowed": True,
    "trained_or_ssl_encoder_cache_scope": "seed_and_fold_training_people_only",
    "reference_units_coordinates": "lock_author_version_contract_before_new_validation_or_test_scores",
    "coordinate_claim": "sourced_nominal_electrode_coordinates_not_measured_person_coordinates",
    "resampling": "resample_physical_signal_never_stretch_array",
    "biot_difference_before_per_channel_normalization": True,
    "biot_bipolar_order": list(BIOT_BIPOLAR), "channel_aliases": CHANNEL_ALIASES,
}
READOUT_RULES = {
    "structure": ["author_representation_or_preregistered_token_pool", "LayerNorm_D", "Linear_D_128", "GELU", "Dropout_0.2", "subject_representation_mean", "Linear_128_2"],
    "keep_native_dimension": True, "parameter_counts_reported_per_method": True,
    "loss": "one_unweighted_cross_entropy_per_subject",
    "classes": ["HC", "MDD"], "validation_all_60_windows": True, "test_all_60_windows": True,
    "frozen_batch": {"subjects": 8, "windows_per_subject": 60, "complete_train_traversal_each_epoch": True},
    "raw_training": {"windows_per_subject": 8, "coverage_epochs": 8, "coverage_windows": 60,
                     "deterministic_cyclic_padding": True, "micro_subjects": [1, 2],
                     "effective_subjects": 8, "last_batch_normalized_by_actual_people": True},
    "ssl": {"subject_balanced_windows": True, "target_real_batch": 64,
            "ssp_and_target_ema_and_author_loss_required": True, "fixed_validation_masks": True,
            "validation_no_gradient_or_ema_updates": True, "ema_once_per_logical_optimizer_step": True,
            "microbatch_loss_accumulation_is_not_vicreg_batch_equivalent": True,
            "author_non_optimizer_hyperparameters": "pending_exact_version_lock"},
}
EVALUATION_RULES = {
    "test_frozen_until_all_formal_fits_complete": True,
    "mdd_probability_threshold": .5, "threshold_tie_class": "HC",
    "primary": "two_class_macro_f1_mean_of_five_folds_then_three_seeds",
    "also_report": ["37_person_oof_per_seed", "balanced_accuracy", "mdd_recall", "auc", "nll", "brier", "confusion_matrix"],
    "111_seed_predictions_are_not_111_independent_people": True,
    "validation_checkpoint_metric": "subject_mean_ce", "all_inference_eval_mode": True,
    "test_checkpoint_selection": False, "learning_rate_grid": False, "threshold_search": False,
    "fusion_weight_selection": False, "extra_14_people_are_not_a_new_blind_test": True,
    "retention_rule": {"macro_f1_delta_min": .01, "strictly_improved_seeds_min": 2,
                       "ba_must_not_decrease": True, "mdd_recall_max_drop": .05,
                       "not_a_significance_test": True, "negative_results_still_complete": True},
}
RECOVERY_RULES = {
    "each_epoch_save": ["model", "optimizer", "scheduler", "rng", "sampler_position", "best_state"],
    "atomic_write_retry": "retry_same_serialized_bytes_no_forward_or_optimizer_repetition",
    "diagnostic_snapshots": list(SNAPSHOT_EPOCHS), "diagnostic_main_comparison": [60, 300],
    "diagnostic_epoch_eval_restores_rng_and_uses_independent_iterator": True,
    "diagnostic_old_first_60_epochs_replay_required": True,
    "finetune_start": "original_official_weights_and_fixed_fresh_head_not_previous_validation_best",
    "engineering_failure_restart_cost_separate": True, "one_gpu_sequential_queue": True,
    "precision": "prefer_bfloat16_else_float32_registered_per_operator",
    "batchnorm_microbatch_statistics_and_true_microbatch_reported": True,
}


def configuration() -> dict:
    """JSON-serializable static protocol, without runtime or test results."""
    return {"experiment_id": EXPERIMENT_ID, "methods": list(METHODS), "seeds": list(SEEDS),
            "folds": list(FOLDS), "conditions": [asdict(c) for c in CONDITIONS],
            "training_policies": {k: asdict(v) for k, v in POLICIES.items()},
            "data": DATA_RULES, "readout": READOUT_RULES, "evaluation": EVALUATION_RULES,
            "recovery": RECOVERY_RULES, "unfreeze_module_rules": UNFREEZE_RULES,
            "formal_fits": EXPECTED_FORMAL_FITS, "supervised_fits": EXPECTED_SUPERVISED_FITS,
            "ssl_fits": EXPECTED_SSL_FITS, "extension_is_same_fit": True,
            "engineering_trials_count_as_paper_completion": False}


def best_validation_epoch(losses: Sequence[float], minimum_improvement: float = 1e-4) -> int:
    """One-based effective-improvement checkpoint; ties retain earlier epoch."""
    if not losses or not all(math.isfinite(float(x)) and x >= 0 for x in losses):
        raise ValueError("Validation losses must be finite nonnegative and nonempty")
    best_epoch, best_loss = 1, float(losses[0])
    for epoch, loss in enumerate(losses[1:], 2):
        if best_loss - float(loss) >= minimum_improvement:
            best_epoch, best_loss = epoch, float(loss)
    return best_epoch


def training_decision(policy_key: str, validation_losses: Sequence[float], *, extended: bool = False) -> dict:
    """Make the registered stop/extension decision from this fit's val only."""
    policy = POLICIES[policy_key]
    epoch = len(validation_losses)
    best = best_validation_epoch(validation_losses, policy.minimum_improvement)
    cap = policy.final_epochs if extended else policy.initial_epochs
    if epoch > cap:
        raise ValueError("History exceeds the current registered epoch cap")
    decision = {"epoch": epoch, "best_epoch": best, "stop": False, "extend": False, "reason": "continue"}
    if not policy.early_stopping:
        if epoch == cap:
            decision.update(stop=True, reason="fixed_epoch_300")
        return decision
    if epoch >= policy.minimum_stop_epoch and epoch - best >= policy.patience:
        decision.update(stop=True, reason="validation_patience")
    elif epoch == cap:
        recent, previous = mean(validation_losses[-20:]), mean(validation_losses[-40:-20])
        improvement = (previous - recent) / previous if previous > 0 else 0.0
        eligible = (not extended and policy.final_epochs > policy.initial_epochs
                    and best > .8 * policy.initial_epochs and improvement >= .01)
        decision.update(stop=not eligible, extend=eligible,
                        reason="registered_extension" if eligible else "epoch_cap",
                        last_20_relative_validation_improvement=improvement)
    return decision


def learning_rate_multiplier(policy_key: str, epoch: int, *, parameter_group: str = "head") -> float:
    """One-based epoch multiplier, holding terminal LR after an extension."""
    policy = POLICIES[policy_key]
    if not 1 <= epoch <= policy.final_epochs or parameter_group not in ("head", "backbone"):
        raise ValueError("Invalid epoch or parameter group")
    if policy.scheduler == "none":
        return 1.0
    start = policy.warmup_epochs if parameter_group == "backbone" and policy.backbone_warmup_epochs else 0
    warmup = policy.backbone_warmup_epochs if start else policy.warmup_epochs
    if epoch <= start:
        return 0.0
    if epoch <= start + warmup:
        return (epoch - start) / warmup
    if epoch >= policy.initial_epochs:
        return .1
    fraction = (epoch - start - warmup) / (policy.initial_epochs - start - warmup)
    return .1 + .9 * .5 * (1. + math.cos(math.pi * fraction))


def subject_cross_entropy(logits: Sequence[Sequence[float]], labels: Sequence[int], subject_ids: Sequence[str]) -> float:
    """Reference metric: exactly one binary logit vector per unique subject.

    The trainable head computes its differentiable analogue after aggregating
    each subject's representations. This helper is for independent metric QA.
    """
    if not logits or len(logits) != len(labels) or len(labels) != len(subject_ids) or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("One row per unique subject is required")
    losses = []
    for row, label in zip(logits, labels):
        if len(row) != 2 or label not in (0, 1) or not all(math.isfinite(float(x)) for x in row):
            raise ValueError("Expected finite [HC, MDD] logits and a binary target")
        maximum = max(row)
        log_z = maximum + math.log(sum(math.exp(float(x) - maximum) for x in row))
        losses.append(log_z - float(row[label]))
    return mean(losses)


if len(CONDITION_BY_ID) != 24 or sum(c.objective == "supervised" for c in CONDITIONS) * len(SEEDS) * len(FOLDS) != 345:
    raise RuntimeError("Formal condition arithmetic differs from the signed plan")
