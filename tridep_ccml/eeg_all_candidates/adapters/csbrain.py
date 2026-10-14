"""Strict official CSBrain backbone with a declared MODMA19 input adapter.

FCz/CPz retain their actual identity and belong to the author's central
region, as specified in model_for_bciciv2a.py. Token mean pooling replaces
the author's large downstream flatten head. This is an adaptation, not an
identical reproduction of the Mumtaz preprocessing/reference pipeline.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import protocol as P
from .. import registry as R

VENDOR = P.WORK / "tridep_ccml/eeg_all_candidates/vendor/csbrain"
ASSETS = P.OUT / "assets"
DIMENSION = 200
REGION_TO_CHANNELS = {
    0: ("FP1", "F7", "F3", "F4", "F8", "FP2"),
    1: ("P3", "Pz", "P4"),
    2: ("T5", "T3", "T4", "T6"),
    3: ("O1", "O2"),
    4: ("FCz", "C3", "C4", "CPz"),
}
REGIONS = tuple(next(region for region, names in REGION_TO_CHANNELS.items() if channel in names) for channel in P.CHANNELS)
SORTED_INDICES = tuple(P.CHANNELS.index(channel) for names in REGION_TO_CHANNELS.values() for channel in names)
INPUT_CONTRACT = {
    "source_shape": ["W", 19, 1000], "encoder_shape": ["W", 19, 5, 200],
    "sample_rate_hz": 200, "seconds_per_window": 5, "native_dimension": 200,
    "source_channel_names": list(P.CHANNELS), "brain_region_ids_original_order": list(REGIONS),
    "sorted_channel_names": [P.CHANNELS[i] for i in SORTED_INDICES],
    "scale": "E2v2 native_numeric_as_EEGLAB_microvolt_reader_convention divided by fixed 100",
    "physical_amplitude_calibration_confirmed": False, "new_statistical_normalization": None,
    "reference": "preserve E2v2 source common reference; author Mumtaz uses left-ear reference",
    "author_reference_reproduced": False,
    "filtering": "retain already boundary-repaired E2v2 filters; no new within-window filtering",
    "pooling": "mean across all channel and one-second time tokens before reconstruction projector",
    "adaptation_differences": ["19 electrode set includes FCz and CPz instead of Fz/Cz", "E2v2 common reference and preprocessing retained", "small head replaces full flatten classifier"],
}


def _source_record(record: dict) -> dict:
    value = {"path": record["path"], "sha256": record["sha256"]}
    R.checked(value)
    return value


def _verified_assets() -> tuple[Path, list[dict], dict]:
    manifest = R.read_json(ASSETS / "download_manifest.json")
    binding = R.read_json(VENDOR / "author_source_binding.json")
    if binding["revision"] != manifest["sources"]["csbrain"]["revision"]:
        raise ValueError("CSBrain vendor revision differs from download registration")
    needed = ("models/CSBrain.py", "models/CSBrain_transformer.py", "models/CSBrain_transformerlayer.py", "models/model_for_mumtaz.py", "models/model_for_bciciv2a.py", "datasets/mumtaz_dataset.py")
    records = {item["path"]: item for item in binding["files"]}
    sources = [_source_record(records[(VENDOR / name).relative_to(P.WORK).as_posix()]) for name in needed]
    sources.append(R.descriptor(VENDOR / "author_source_binding.json"))
    weight = next(item for item in manifest["weights"]["csbrain"]["files"] if item["path"].endswith("/CSBrain.pth"))
    return R.checked(_source_record(weight)), sources, {"repository": binding["repository"], "revision": binding["revision"], "weight": _source_record(weight)}


def _author_class():
    """Load unchanged author files without leaking their generic models alias."""
    isolated_name = "_eeg_all_candidates_csbrain_models"
    module_name = isolated_name + ".CSBrain"
    if module_name in sys.modules:
        return sys.modules[module_name].CSBrain
    package = types.ModuleType(isolated_name)
    package.__path__ = [str(VENDOR / "models")]
    sys.modules[isolated_name] = package
    previous = {key: value for key, value in sys.modules.items() if key == "models" or key.startswith("models.")}
    for key in previous:
        del sys.modules[key]
    sys.modules["models"] = package
    try:
        spec = importlib.util.spec_from_file_location(module_name, VENDOR / "models/CSBrain.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        for key, value in list(sys.modules.items()):
            if key.startswith("models."):
                sys.modules[isolated_name + key[len("models"):]] = value
        return module.CSBrain
    finally:
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                del sys.modules[key]
        sys.modules.update(previous)


class CSBrainEncoder(nn.Module):
    """Keep frozen dropout/module state in eval during limited fine-tuning."""
    output_dimension = DIMENSION

    def __init__(self, backbone: nn.Module, trainable: bool):
        super().__init__()
        self.backbone = backbone
        self.limited_finetune = trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if trainable:
            for parameter in self.backbone.encoder.layers[-1].parameters():
                parameter.requires_grad_(True)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.limited_finetune:
            self.backbone.encoder.layers[-1].train(True)
        return self

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4 or tuple(windows.shape[1:]) != (19, 5, 200):
            raise ValueError("CSBrain expects adapted [W,19,5,200] windows")
        tokens = self.backbone(windows)
        if tuple(tokens.shape[1:]) != (19, 5, DIMENSION):
            raise ValueError("CSBrain author token output differs from registered interface")
        return tokens.mean(dim=(1, 2))


def load_encoder(device: str = "cuda", trainable: bool = False) -> tuple[CSBrainEncoder, dict]:
    """Load every formal backbone key strictly, then remove reconstruction head."""
    weight, sources, provenance = _verified_assets()
    backbone = _author_class()(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                               seq_len=30, n_layer=12, nhead=8,
                               brain_regions=list(REGIONS), sorted_indices=list(SORTED_INDICES))
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected CSBrain formal plain backbone state dictionary")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    backbone.load_state_dict(state, strict=True)
    if not all(torch.equal(backbone.state_dict()[key], value) for key, value in state.items()):
        raise ValueError("CSBrain loaded state is not byte-identical to formal weights")
    full_parameters = sum(parameter.numel() for parameter in backbone.parameters())
    removed = [key for key in state if key.startswith("proj_out.")]
    # This matches the author downstream representation entry, before its
    # large flatten classifier, while still auditing full pretrained loading.
    backbone.proj_out = nn.Identity()
    model = CSBrainEncoder(backbone, trainable).to(device)
    audit = {"method": "CSBrain", "formal_pretrained_strict_loaded": True,
             "source": provenance, "implementation_sources": sources + [R.descriptor(Path(__file__))],
             "weight_sources": [provenance["weight"]], "input_contract": INPUT_CONTRACT,
             "loaded_backbone_keys": len(state), "full_pretrained_parameters": full_parameters,
             "removed_reconstruction_projector_keys": removed,
             "retained_encoder_parameters": sum(p.numel() for p in model.parameters()),
             "native_dimension": DIMENSION, "trainable_parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
             "frozen_parameter_names": [n for n, p in model.named_parameters() if not p.requires_grad],
             "last_block_prefix": "backbone.encoder.layers.11.",
             "output_normalization": "no separate backbone output normalization; final block norm1/norm2/norm3 belong to last block; fresh head LayerNorm is separate",
             "engineering_status": "strict_loading_checked_forward_gradient_resume_pending",
             "gpu_used_by_adapter_load": str(device).startswith("cuda")}
    return model, audit


def adapt_raw(raw: np.ndarray) -> np.ndarray:
    """Apply fixed author downstream numeric scaling; fit no cohort statistics."""
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 3 or tuple(raw.shape[1:]) != (19, 1000) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite np.float32 raw [W,19,1000] in registered channel order")
    return np.ascontiguousarray((raw / np.float32(100.)).reshape(-1, 19, 5, 200))


def encode_windows(model: CSBrainEncoder, tensor: torch.Tensor) -> torch.Tensor:
    """Return native-D window representations; caller controls grad/inference."""
    return model(tensor)
