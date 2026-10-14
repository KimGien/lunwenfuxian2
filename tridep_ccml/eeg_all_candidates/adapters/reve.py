"""Official REVE Base and official nominal position-bank adaptation.

The Hugging Face code shipped at the exact checkpoint revision is loaded
locally, without network execution or partial-key loading. Author XYZ values
are kept at their native metre scale and FCz/CPz use their own bank entries.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn

from .. import protocol as P
from .. import registry as R

ASSETS = P.OUT / "assets"
VENDOR = P.WORK / "tridep_ccml/eeg_all_candidates/vendor/reve"
DIMENSION = 512
INPUT_CONTRACT = {
    "source_shape": ["W", 19, 1000], "encoder_shape": ["W", 19, 1000],
    "sample_rate_hz": 200, "seconds_per_window": 5, "patch_samples": 200,
    "patch_overlap_samples": 20, "patch_stride_samples": 180,
    "patches_per_channel": 5, "patch_start_samples": [0, 180, 360, 540, 720],
    "last_used_sample_exclusive": 920, "trailing_samples_unused_by_author_unfold": 80,
    "source_channel_names": list(P.CHANNELS), "native_dimension": DIMENSION,
    "scale": "E2v2 native_numeric_as_EEGLAB_microvolt_reader_convention divided by fixed 100 as author Mumtaz LMDBDataset",
    "physical_amplitude_calibration_confirmed": False, "new_statistical_normalization": None,
    "reference": "preserve E2v2 source common reference; author Mumtaz uses left-ear reference",
    "author_reference_reproduced": False,
    "filtering": "retain already boundary-repaired E2v2 filters; no new within-window filtering",
    "pooling": "registered mean of final-layer channel/time tokens, retaining all 512 dimensions",
    "position_source": "official brain-bzh/reve-positions nominal standard1005 bank",
    "position_units": "metres", "position_frame": "nominal standard1005 MRI XYZ frame",
    "position_transform": "none; preserve author's numeric XYZ values",
    "positions_are_measured_from_subject": False,
    "adaptation_differences": ["E2v2 reference and existing filtering retained", "native token mean pooling plus common small head", "fixed downstream /100 differs from pretraining recording z-score; follows author Mumtaz task"],
}


def _source_record(record: dict) -> dict:
    value = {"path": record["path"], "sha256": record["sha256"]}
    R.checked(value)
    return value


def _asset_files(name: str) -> tuple[dict, list[dict]]:
    manifest = R.read_json(ASSETS / "download_manifest.json")
    entry = manifest["weights"][name]
    sources = [_source_record(record) for record in entry["files"]]
    files = {Path(record["path"]).name: R.checked(record) for record in sources}
    return {"repository": entry["repository"], "revision": entry["revision"], "files": files}, sources


def nominal_positions() -> tuple[torch.Tensor, dict]:
    """Strict channel coverage and cross-check against author's JSON and MNE."""
    provenance, sources = _asset_files("reve_positions")
    files = provenance["files"]
    config = R.read_json(files["config.json"])
    names = config["position_names"]
    if len(names) != len(set(names)) or any(name not in names for name in P.CHANNELS):
        raise ValueError("Official REVE position bank does not cover all exact channel names")
    state = load_file(str(files["model.safetensors"]), device="cpu")
    if set(state) != {"embedding"} or tuple(state["embedding"].shape) != (len(names), 3):
        raise ValueError("Official position-bank state shape differs")
    indices = [names.index(name) for name in P.CHANNELS]
    positions = state["embedding"][indices].clone()
    coordinates = R.read_json(files["positions.json"])
    registered = torch.tensor([coordinates[name] for name in P.CHANNELS], dtype=torch.float32)
    if not torch.equal(positions, registered) or not torch.isfinite(positions).all():
        raise ValueError("Position JSON and official learned-bank state disagree")
    # The official bank is nominal montage coordinates, not participant data.
    # This check establishes units/frame instead of inferring them from range.
    import mne
    standard = mne.channels.make_standard_montage("standard_1005").get_positions()
    match = {name.lower(): value for name, value in standard["ch_pos"].items()}
    expected = np.asarray([match[name.lower()] for name in P.CHANNELS])
    maximum_error = float(np.max(np.abs(expected - positions.numpy())))
    if standard["coord_frame"] != "mri" or maximum_error > 1e-7:
        raise ValueError("REVE bank no longer matches nominal standard1005 metre/MRI coordinates")
    return positions, {"repository": provenance["repository"], "revision": provenance["revision"],
                       "sources": sources, "channel_names": list(P.CHANNELS),
                       "xyz": positions.tolist(), "units": "metres", "frame": "mri",
                       "mne_standard1005_max_abs_error": maximum_error, "mne_version": mne.__version__,
                       "source_is_nominal_montage_not_measured_subject_positions": True}


def _author_class(files: dict):
    """Import the checkpoint's unchanged local custom modules in a private package."""
    name = "_eeg_all_candidates_reve_checkpoint"
    if name + ".modeling_reve" not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(files["modeling_reve.py"].parent)]
        sys.modules[name] = package
        for suffix in ("configuration_reve", "modeling_reve"):
            module_name = name + "." + suffix
            spec = importlib.util.spec_from_file_location(module_name, files[suffix + ".py"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    return sys.modules[name + ".configuration_reve"].ReveConfig, sys.modules[name + ".modeling_reve"].Reve


class ReveEncoder(nn.Module):
    output_dimension = DIMENSION

    def __init__(self, backbone: nn.Module, positions: torch.Tensor, trainable: bool):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("positions", positions)
        self.limited_finetune = trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if trainable:
            for parameter in self.backbone.transformer.layers[-1].parameters():
                parameter.requires_grad_(True)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.limited_finetune:
            self.backbone.transformer.layers[-1].train(True)
        return self

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (19, 1000):
            raise ValueError("REVE expects adapted [W,19,1000] windows")
        positions = self.positions.unsqueeze(0).expand(len(windows), -1, -1)
        tokens = self.backbone(windows, positions, return_output=False)
        if tuple(tokens.shape[1:]) != (19, 5, DIMENSION):
            raise ValueError("REVE author token interface differs from registration")
        return tokens.mean(dim=(1, 2))


def load_encoder(device: str = "cuda", trainable: bool = False) -> tuple[ReveEncoder, dict]:
    provenance, sources = _asset_files("reve")
    files = provenance["files"]
    config_class, encoder_class = _author_class(files)
    config_data = R.read_json(files["config.json"])
    if any(config_data[key] != value for key, value in {"embed_dim": 512, "depth": 22, "heads": 8, "patch_size": 200, "patch_overlap": 20}.items()):
        raise ValueError("Expected the registered REVE Base architecture")
    config = config_class.from_dict(config_data)
    backbone = encoder_class(config)
    state = load_file(str(files["model.safetensors"]), device="cpu")
    backbone.load_state_dict(state, strict=True)
    if not all(torch.equal(backbone.state_dict()[key], value) for key, value in state.items()):
        raise ValueError("REVE formal pretrained weights did not load identically")
    positions, position_audit = nominal_positions()
    model = ReveEncoder(backbone, positions, trainable).to(device)
    binding = R.read_json(VENDOR / "author_source_binding.json")
    needed = ("preprocessing/preprocessing_mumtaz.py", "src/downstream_tasks/dataloaders.py", "src/configs/task/mumtaz.yaml", "src/models/classifier.py")
    records = {item["path"]: item for item in binding["files"]}
    preprocessing_sources = [_source_record(records[(VENDOR / name).relative_to(P.WORK).as_posix()]) for name in needed]
    audit = {"method": "REVE", "formal_pretrained_strict_loaded": True,
             "repository": provenance["repository"], "revision": provenance["revision"],
             "implementation_sources": [source for source in sources if not source["path"].endswith("model.safetensors")]
                                       + preprocessing_sources + [R.descriptor(Path(__file__))],
             "weight_sources": [source for source in sources if source["path"].endswith("model.safetensors")],
             "position_bank": position_audit, "input_contract": INPUT_CONTRACT,
             "loaded_backbone_keys": len(state), "encoder_parameters": sum(p.numel() for p in model.parameters()),
             "native_dimension": DIMENSION, "trainable_parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
             "frozen_parameter_names": [n for n, p in model.named_parameters() if not p.requires_grad],
             "last_block_prefix": "backbone.transformer.layers.21.",
             "output_normalization": "author final_layer is Identity; ln is positional norm, kept frozen; common fresh head LayerNorm supplies output normalization",
             "unused_pretrained_query_token": "backbone.cls_query_token remains frozen; registered token mean pooling does not use it",
             "engineering_status": "strict_loading_and_nominal_positions_checked_forward_gradient_resume_pending",
             "gpu_used_by_adapter_load": str(device).startswith("cuda")}
    return model, audit


def adapt_raw(raw: np.ndarray) -> np.ndarray:
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 3 or tuple(raw.shape[1:]) != (19, 1000) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite np.float32 raw [W,19,1000] in registered channel order")
    return np.ascontiguousarray(raw / np.float32(100.))


def encode_windows(model: ReveEncoder, tensor: torch.Tensor) -> torch.Tensor:
    return model(tensor)
