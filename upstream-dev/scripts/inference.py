"""Train per-fold models and generate per-subject predictions.

Usage
-----
    python scripts/inference.py --config configs/training/text.yaml --name macbert_lstm
    python scripts/inference.py --config configs/training/speech.yaml --name hubert_bigru_conv
    python scripts/inference.py --config configs/training/eeg.yaml --name cbramod_mumtaz_conv
    python scripts/inference.py --config configs/training/early_fusion.yaml --name early_concat
    python scripts/inference.py --config configs/training/intermediate_fusion.yaml --name intermediate_concat
    python scripts/inference.py --config configs/training/text.yaml --all
    python scripts/inference.py --config configs/training/eeg.yaml --list

CLI overrides (OmegaConf dot-notation):
    python scripts/inference.py --config configs/training/eeg.yaml --all data.base_dir=/custom/path
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import StandardScaler

from lib.datasets import (
    RawAudioDataset,
    TextDataset,
    TrimodalDataset,
    collate_fn_with_ids,
    collate_raw_audio,
    collate_trimodal,
)
from lib.models import (
    BiLSTMClassifier,
    ConvPoolClassifier,
    EarlyFusionBottleneck,
    EarlyFusionConcat,
    EEGCNNFCClassifier,
    EEGCNNGRUAttentionClassifier,
    EEGCNNLSTMClassifier,
    EEGConvPoolClassifier,
    EEGLSTMClassifier,
    GRUAttentionClassifier,
    IntermediateFusionConcat,
    IntermediateFusionGated,
    LSTM1Classifier,
    ProsodyMFCCModel,
    SpeechConvPoolClassifier,
)

DATA_DIR = PROJECT_ROOT / "data" / "split_dataset_june"
FOLD_FILE = DATA_DIR / "fold_assignments.json"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
PREDICTIONS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

MODEL_REGISTRY = {
    "LSTM1Classifier": LSTM1Classifier,
    "ConvPoolClassifier": ConvPoolClassifier,
    "BiLSTMClassifier": BiLSTMClassifier,
    "SpeechConvPoolClassifier": SpeechConvPoolClassifier,
    "GRUAttentionClassifier": GRUAttentionClassifier,
    "EEGLSTMClassifier": EEGLSTMClassifier,
    "EEGConvPoolClassifier": EEGConvPoolClassifier,
    "EEGCNNLSTMClassifier": EEGCNNLSTMClassifier,
    "EEGCNNFCClassifier": EEGCNNFCClassifier,
    "EEGCNNGRUAttentionClassifier": EEGCNNGRUAttentionClassifier,
    "ProsodyMFCCModel": ProsodyMFCCModel,
    "EarlyFusionConcat": EarlyFusionConcat,
    "EarlyFusionBottleneck": EarlyFusionBottleneck,
    "IntermediateFusionConcat": IntermediateFusionConcat,
    "IntermediateFusionGated": IntermediateFusionGated,
}

UNIMODAL_MODEL_CLS = {
    "EEGConvPoolClassifier": EEGConvPoolClassifier,
    "SpeechConvPoolClassifier": SpeechConvPoolClassifier,
    "LSTM1Classifier": LSTM1Classifier,
}


def load_unimodal_model(
    modality_info: dict,
    fold_idx: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load a pretrained unimodal checkpoint for one cross-validation fold.

    Args:
        modality_info: Dict with keys ``"model_class"``, ``"model_kwargs"``,
            and ``"ckpt_dir"`` (relative to ``CHECKPOINTS_DIR``).
        fold_idx: Zero-based fold index; the file ``fold_{fold_idx+1}.pt``
            is loaded.
        device: Target device for the loaded model.

    Returns:
        Model in eval mode with loaded weights on *device*.
    """
    cls = UNIMODAL_MODEL_CLS[modality_info["model_class"]]
    model = cls(**modality_info["model_kwargs"])
    ckpt_path = CHECKPOINTS_DIR / modality_info["ckpt_dir"] / f"fold_{fold_idx + 1}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def get_device() -> torch.device:
    """Select the best available compute device.

    Returns:
        ``cuda`` if a GPU is available, ``mps`` on Apple Silicon, otherwise
        ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_all_segments(dataset: "RawAudioDataset") -> np.ndarray:
    """Concatenate all raw audio segments from a dataset for scaler fitting.

    Args:
        dataset: A :class:`~lib.datasets.RawAudioDataset` instance.

    Returns:
        2-D array of shape ``(total_segments, feat_dim)`` containing every
        segment from every subject in the dataset.
    """
    all_segments = []
    for i in range(len(dataset)):
        segments, _, _ = dataset[i]
        for seg in segments:
            all_segments.append(seg.numpy())
    return np.concatenate(all_segments, axis=0)


def normalize_segments(
    segments_list: list,
    scaler: "StandardScaler",
) -> list:
    """Apply a fitted :class:`~sklearn.preprocessing.StandardScaler` to a batch.

    Args:
        segments_list: List of per-subject segment lists, where each inner
            list contains :class:`~torch.Tensor` objects of shape
            ``(n_frames, feat_dim)``.
        scaler: A fitted ``StandardScaler`` instance.

    Returns:
        Nested list with the same structure as *segments_list* but with each
        segment standardised and returned as a ``torch.float32`` tensor.
    """
    normalized = []
    for subject_seqs in segments_list:
        norm_subject = []
        for seg in subject_seqs:
            arr = scaler.transform(seg.numpy())
            norm_subject.append(torch.tensor(arr, dtype=torch.float32))
        normalized.append(norm_subject)
    return normalized


def train_model(
    model: torch.nn.Module,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    segment_level: bool = False,
    raw_audio: bool = False,
    scaler: "StandardScaler | None" = None,
    early_fusion: bool = False,
    intermediate_fusion: bool = False,
) -> torch.nn.Module:
    """Train a model for a fixed number of epochs and return it.

    Supports four distinct forward-pass modes selected by the boolean flags:

    * **trimodal** (``early_fusion`` or ``intermediate_fusion``) — batch
      contains ``(eeg, speech, text, y, subject_ids)``; calls
      ``model(eeg, speech, text)``.
    * **raw_audio** — batch contains ``(segments_list, y, subject_ids)``; each
      subject's segments are normalised with *scaler* before the forward pass.
    * **segment_level** — batch ``x`` has shape
      ``(batch, n_segments, C, F)``; segments are flattened and labels
      are expanded so the model is trained segment-by-segment.
    * **default** — standard ``(x, y, subject_ids)`` batch.

    The validation loader is not used for early stopping; it is accepted
    only for API consistency.

    Args:
        model: Uninitialised or pretrained model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split (unused internally).
        criterion: Loss function (e.g. ``nn.CrossEntropyLoss``).
        optimizer: Optimiser instance already constructed for *model*.
        device: Torch device to move tensors to.
        epochs: Number of full passes over the training data.
        segment_level: If ``True``, treat each segment independently during
            training (EEG CNN classifiers).
        raw_audio: If ``True``, batches contain variable-length segment
            lists instead of fixed tensors.
        scaler: Fitted ``StandardScaler`` required when *raw_audio* is
            ``True``.
        early_fusion: If ``True``, use trimodal forward pass.
        intermediate_fusion: If ``True``, use trimodal forward pass.

    Returns:
        The trained model (same object as *model*).
    """
    trimodal = early_fusion or intermediate_fusion
    for _ in range(epochs):
        model.train()
        if trimodal:
            for eeg, speech, text, y, _ in train_loader:
                eeg, speech, text, y = (
                    eeg.to(device),
                    speech.to(device),
                    text.to(device),
                    y.to(device),
                )
                optimizer.zero_grad()
                criterion(model(eeg, speech, text), y).backward()
                optimizer.step()
        elif raw_audio:
            for segments_list, y, _ in train_loader:
                segments_list = normalize_segments(segments_list, scaler)
                y = y.to(device)
                optimizer.zero_grad()
                criterion(model(segments_list), y).backward()
                optimizer.step()
        else:
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                if segment_level:
                    # x: (batch, n_segments, C, F) -> (batch*n_segments, C, F)
                    b, s = x.shape[0], x.shape[1]
                    x = x.reshape(b * s, *x.shape[2:])
                    y = y.unsqueeze(1).expand(-1, s).reshape(b * s)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
    return model


def run_config(cfg: dict, device: torch.device) -> tuple[float, float]:
    """Train a single configuration over 5 folds and save predictions.

    Reads the fold assignment file, builds datasets and data loaders,
    trains one model per fold, evaluates on the test split, and writes a
    unified predictions CSV to ``predictions/<cfg['output_csv']>``.

    Args:
        cfg: Experiment configuration dict with at minimum the keys
            ``name``, ``model_class``, ``model_kwargs``, ``epochs``,
            ``lr``, ``batch_size``, and ``output_csv``.  Fusion configs
            additionally need ``eeg_file``, ``speech_file``, ``text_file``
            (multimodal), or ``embedding_file`` (unimodal / raw_audio).
        device: Torch device for training and inference.

    Returns:
        Tuple ``(mean_f1, std_f1)`` — mean and sample standard deviation of
        per-fold macro-F1 scores.
    """
    print(f"\n{'=' * 70}")
    print(f"Config : {cfg['name']}")
    if cfg.get("early_fusion") or cfg.get("intermediate_fusion"):
        print(f"  eeg    : {cfg['eeg_file']}")
        print(f"  speech : {cfg['speech_file']}")
        print(f"  text   : {cfg['text_file']}")
    else:
        print(f"  embed  : {cfg['embedding_file']}")
    print(f"  model  : {cfg['model_class']}  {cfg['model_kwargs']}")
    print(f"  epochs={cfg['epochs']}  lr={cfg['lr']}  batch={cfg['batch_size']}")
    print(f"{'=' * 70}")

    set_seed(42)

    with open(FOLD_FILE) as f:
        folds = json.load(f)

    cfg_ckpt_dir = CHECKPOINTS_DIR / cfg["name"]
    cfg_ckpt_dir.mkdir(exist_ok=True)

    all_predictions: list[dict] = []
    fold_f1_scores: list[float] = []
    segment_level = cfg["model_class"] in (
        "EEGCNNLSTMClassifier",
        "EEGCNNFCClassifier",
        "EEGCNNGRUAttentionClassifier",
    )
    raw_audio = cfg.get("raw_audio", False)
    early_fusion = cfg.get("early_fusion", False)
    intermediate_fusion = cfg.get("intermediate_fusion", False)
    trimodal = early_fusion or intermediate_fusion

    for fold_idx in range(5):
        fold_name = f"fold_{fold_idx + 1}"
        train_ids = folds[fold_name]["train"]
        val_ids = folds[fold_name]["val"]
        test_ids = folds[fold_name]["test"]

        mk_dirs = lambda ids: [str(DATA_DIR / sid) for sid in ids]

        if trimodal:
            ef_args = (cfg["eeg_file"], cfg["speech_file"], cfg["text_file"])
            train_set = TrimodalDataset(mk_dirs(train_ids), *ef_args)
            val_set = TrimodalDataset(mk_dirs(val_ids), *ef_args)
            test_set = TrimodalDataset(mk_dirs(test_ids), *ef_args)
            collate = collate_trimodal
        elif raw_audio:
            train_set = RawAudioDataset(mk_dirs(train_ids), cfg["embedding_file"])
            val_set = RawAudioDataset(mk_dirs(val_ids), cfg["embedding_file"])
            test_set = RawAudioDataset(mk_dirs(test_ids), cfg["embedding_file"])
            collate = collate_raw_audio
        else:
            train_set = TextDataset(mk_dirs(train_ids), cfg["embedding_file"])
            val_set = TextDataset(mk_dirs(val_ids), cfg["embedding_file"])
            test_set = TextDataset(mk_dirs(test_ids), cfg["embedding_file"])
            collate = collate_fn_with_ids

        bs = cfg["batch_size"]
        train_loader = DataLoader(
            train_set, batch_size=bs, shuffle=True, collate_fn=collate
        )
        val_loader = DataLoader(
            val_set, batch_size=bs, shuffle=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_set, batch_size=bs, shuffle=False, collate_fn=collate
        )

        # Fit scaler on training data for raw_audio configs
        scaler = None
        if raw_audio:
            print(f"  Fold {fold_idx + 1}  fitting StandardScaler on training data...")
            train_data = extract_all_segments(train_set)
            scaler = StandardScaler()
            scaler.fit(train_data)

        if intermediate_fusion:
            eeg_model = load_unimodal_model(
                cfg["_unimodal_checkpoints"]["eeg"], fold_idx, device
            )
            speech_model = load_unimodal_model(
                cfg["_unimodal_checkpoints"]["speech"], fold_idx, device
            )
            text_model = load_unimodal_model(
                cfg["_unimodal_checkpoints"]["text"], fold_idx, device
            )
            model = MODEL_REGISTRY[cfg["model_class"]](
                eeg_model=eeg_model,
                speech_model=speech_model,
                text_model=text_model,
                **cfg["model_kwargs"],
            ).to(device)
        else:
            model = MODEL_REGISTRY[cfg["model_class"]](**cfg["model_kwargs"]).to(device)

        # For intermediate fusion, only optimize the trainable fusion head
        params_to_optimize = (
            [p for p in model.parameters() if p.requires_grad]
            if intermediate_fusion
            else model.parameters()
        )

        opt_name = cfg.get("optimizer", "Adamax")
        opt_cls = {
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "Adamax": optim.Adamax,
        }.get(opt_name, optim.Adamax)
        optimizer = opt_cls(
            params_to_optimize, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0)
        )
        criterion = nn.CrossEntropyLoss()

        trained = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            cfg["epochs"],
            segment_level=segment_level,
            raw_audio=raw_audio,
            scaler=scaler,
            early_fusion=early_fusion,
            intermediate_fusion=intermediate_fusion,
        )

        ckpt_path = cfg_ckpt_dir / f"fold_{fold_idx + 1}.pt"
        torch.save(trained.state_dict(), ckpt_path)
        print(f"  Fold {fold_idx + 1}  checkpoint -> {ckpt_path}")

        trained.eval()
        fold_preds, fold_labels = [], []

        with torch.no_grad():
            if trimodal:
                for eeg, speech, text, y, subject_ids in test_loader:
                    eeg = eeg.to(device)
                    speech = speech.to(device)
                    text = text.to(device)
                    logits = trained(eeg, speech, text)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = logits.argmax(dim=1).cpu().numpy()
                    labels = y.numpy()
                    fold_preds.extend(preds)
                    fold_labels.extend(labels)
                    for sid, pred, prob, gt in zip(subject_ids, preds, probs, labels):
                        all_predictions.append(
                            {
                                "subject_id": sid,
                                "fold": fold_idx + 1,
                                "ground_truth": int(gt),
                                "predicted_label": int(pred),
                                "probability": float(prob),
                            }
                        )

            if not trimodal:
                for batch_x, y, subject_ids in test_loader:
                    if raw_audio:
                        batch_x = normalize_segments(batch_x, scaler)
                        logits = trained(batch_x)
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        preds = logits.argmax(dim=1).cpu().numpy()
                        labels = y.numpy()
                        fold_preds.extend(preds)
                        fold_labels.extend(labels)
                        for sid, pred, prob, gt in zip(
                            subject_ids, preds, probs, labels
                        ):
                            all_predictions.append(
                                {
                                    "subject_id": sid,
                                    "fold": fold_idx + 1,
                                    "ground_truth": int(gt),
                                    "predicted_label": int(pred),
                                    "probability": float(prob),
                                }
                            )
                    elif segment_level:
                        # Each subject's embedding is (n_segments, 29, 10).
                        # Run model on segments, majority-vote per subject.
                        for i, sid in enumerate(subject_ids):
                            segments = batch_x[i].to(device)  # (n_segments, 29, 10)
                            seg_logits = trained(segments)
                            seg_probs = torch.softmax(seg_logits, dim=1)[:, 1].cpu()
                            seg_preds = seg_logits.argmax(dim=1).cpu()
                            subject_pred = int(seg_preds.float().mean().round().item())
                            subject_prob = float(seg_probs.mean().item())
                            gt = int(y[i].item())
                            fold_preds.append(subject_pred)
                            fold_labels.append(gt)
                            all_predictions.append(
                                {
                                    "subject_id": sid,
                                    "fold": fold_idx + 1,
                                    "ground_truth": gt,
                                    "predicted_label": subject_pred,
                                    "probability": subject_prob,
                                }
                            )
                    else:
                        batch_x = batch_x.to(device)
                        logits = trained(batch_x)
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        preds = logits.argmax(dim=1).cpu().numpy()
                        labels = y.numpy()
                        fold_preds.extend(preds)
                        fold_labels.extend(labels)
                        for sid, pred, prob, gt in zip(
                            subject_ids, preds, probs, labels
                        ):
                            all_predictions.append(
                                {
                                    "subject_id": sid,
                                    "fold": fold_idx + 1,
                                    "ground_truth": int(gt),
                                    "predicted_label": int(pred),
                                    "probability": float(prob),
                                }
                            )

        f1 = f1_score(fold_labels, fold_preds, average="macro")
        acc = np.mean(np.array(fold_labels) == np.array(fold_preds))
        fold_f1_scores.append(f1)
        print(f"  Fold {fold_idx + 1}  Acc={acc:.4f}  F1(macro)={f1:.4f}")

    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores, ddof=1)
    print(f"\n  Mean F1 = {mean_f1:.4f} +/- {std_f1:.4f}")

    out_path = PREDICTIONS_DIR / cfg["output_csv"]
    with open(out_path, "w") as f:
        f.write("subject_id,fold,ground_truth,predicted_label,probability\n")
        for row in sorted(all_predictions, key=lambda r: (r["fold"], r["subject_id"])):
            f.write(
                f"{row['subject_id']},{row['fold']},{row['ground_truth']},{row['predicted_label']},{row['probability']:.6f}\n"
            )
    print(f"  Predictions -> {out_path}")

    return mean_f1, std_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train per-fold models and generate per-subject predictions."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a training YAML config (e.g. configs/training/eeg.yaml).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Run a single configuration by name.")
    group.add_argument("--all", action="store_true", help="Run all configs in the file.")
    group.add_argument("--list", action="store_true", help="List available configs and exit.")
    # Remaining args are OmegaConf dot-notation overrides, e.g. data.base_dir=/path
    args, overrides = parser.parse_known_args()

    # Load YAML and apply CLI overrides
    yaml_cfg = OmegaConf.load(args.config)
    if overrides:
        yaml_cfg = OmegaConf.merge(yaml_cfg, OmegaConf.from_dotlist(overrides))

    # Override data paths if specified in yaml
    global DATA_DIR, FOLD_FILE
    if "data" in yaml_cfg:
        DATA_DIR = PROJECT_ROOT / yaml_cfg.data.base_dir
        FOLD_FILE = PROJECT_ROOT / yaml_cfg.data.fold_file

    # Load list of experiment configs
    ALL_CONFIGS = OmegaConf.to_container(yaml_cfg.configs, resolve=True)

    # Inject unimodal_checkpoints into each intermediate fusion config
    if "unimodal_checkpoints" in yaml_cfg:
        uc = OmegaConf.to_container(yaml_cfg.unimodal_checkpoints, resolve=True)
        for cfg in ALL_CONFIGS:
            if cfg.get("intermediate_fusion"):
                cfg["_unimodal_checkpoints"] = uc

    CONFIG_BY_NAME = {cfg["name"]: cfg for cfg in ALL_CONFIGS}

    if args.list:
        for cfg in ALL_CONFIGS:
            group = cfg.get("feature_group", "")
            print(
                f"  {cfg['name']:30s}  {cfg['model_class']:28s}  [{group}]  -> {cfg['output_csv']}"
            )
        print(f"\n  Total: {len(ALL_CONFIGS)} configs")
        return

    device = get_device()
    print(f"Device: {device}")

    configs = ALL_CONFIGS if args.all else [CONFIG_BY_NAME[args.name]]
    results = []
    for cfg in configs:
        mean_f1, std_f1 = run_config(cfg, device)
        results.append((cfg, mean_f1, std_f1))

    if args.all and len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY — All configurations")
        print("=" * 70)
        print(f"{'Config':<30s}  {'Feature Group':<18s}  {'F1 (mean ± std)'}")
        print("-" * 70)
        for cfg, f1, std in sorted(results, key=lambda r: -r[1]):
            group = cfg.get("feature_group", "")
            print(f"  {cfg['name']:<28s}  {group:<18s}  {f1:.4f} ± {std:.4f}")

        # Best per feature group
        print("\n" + "-" * 70)
        print("BEST per feature group:")
        groups = {}
        for cfg, f1, std in results:
            g = cfg.get("feature_group", "unknown")
            if g not in groups or f1 > groups[g][1]:
                groups[g] = (cfg, f1, std)
        for g in sorted(groups):
            cfg, f1, std = groups[g]
            print(f"  {g:<18s}  {cfg['name']:<28s}  {f1:.4f} ± {std:.4f}")

        # Best overall
        best_cfg, best_f1, best_std = max(results, key=lambda r: r[1])
        print(f"\nBEST overall: {best_cfg['name']}  {best_f1:.4f} ± {best_std:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
