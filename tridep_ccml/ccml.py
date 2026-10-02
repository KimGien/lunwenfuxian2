"""TC evidential heads, fusion, and losses.

Inputs use [batch, view, class] and class order [HC, MDD]. Losses sum
views per subject and then average subjects. See references/ccml/source.json
for the pinned upstream source and documented numerical adaptation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODALITIES = ("eeg", "speech", "text")


def _finite_float(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _evidence(value: Tensor, multiview: bool = False) -> None:
    _finite_float(value, "evidence")
    if value.ndim < 2 or value.shape[-1] < 2 or any(n == 0 for n in value.shape):
        raise ValueError("evidence needs nonempty batch and class dimensions")
    if multiview and value.ndim != 3:
        raise ValueError("multiview evidence must have shape [batch, view, class]")
    if (value < 0).any().item():
        raise ValueError("evidence must be nonnegative")


def _targets(labels: Tensor, batch: int, classes: int, device: torch.device) -> Tensor:
    if labels.ndim != 1 or labels.shape[0] != batch or labels.dtype != torch.long:
        raise ValueError("labels must be a torch.long tensor of shape [batch]")
    if labels.device != device:
        raise ValueError("labels and model tensors must use the same device")
    if ((labels < 0) | (labels >= classes)).any().item():
        raise ValueError("labels are outside the class range")
    return F.one_hot(labels, classes)


def _finite_result(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"nonfinite {name}; stop and record the failure")
    return value


class ThreeViewHeads(nn.Module):
    """Independent linear heads; default configuration has 774 parameters.

    forward returns raw logits for the paired CE control. Call evidence for
    Softplus outputs. Modality order is explicit and must accompany saved data.
    """

    def __init__(
        self,
        in_features: int = 128,
        num_classes: int = 2,
        modalities: Sequence[str] = MODALITIES,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        if in_features < 1 or num_classes < 2 or not self.modalities:
            raise ValueError("positive input dimensions and at least two classes required")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique")
        self.in_features = in_features
        self.num_classes = num_classes
        self.heads = nn.ModuleDict({name: nn.Linear(in_features, num_classes) for name in self.modalities})

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        if isinstance(features, Mapping):
            if set(features) != set(self.modalities):
                raise ValueError("feature mapping must match the configured modalities")
            features = torch.stack([features[name] for name in self.modalities], dim=1)
        _finite_float(features, "features")
        if features.ndim != 3 or features.shape[1:] != (len(self.modalities), self.in_features):
            raise ValueError("features must have shape [batch, configured views, in_features]")
        if features.shape[0] == 0:
            raise ValueError("features must have a nonempty batch")
        logits = torch.stack([self.heads[name](features[:, i]) for i, name in enumerate(self.modalities)], dim=1)
        return _finite_result(logits, "head logits")

    def evidence(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        return F.softplus(self(features))


def dirichlet_from_evidence(evidence: Tensor) -> dict[str, Tensor]:
    """Dirichlet mean and evidence insufficiency, never a softmax of evidence.

    The last dimension indexes classes. u has that dimension removed.
    """
    _evidence(evidence)
    alpha = evidence + 1
    strength = _finite_result(alpha.sum(dim=-1, keepdim=True), "Dirichlet strength")
    return {"alpha": alpha, "p": alpha / strength, "u": (alpha.shape[-1] / strength).squeeze(-1)}


def evidence_dc(evidence: Tensor, rho: float = 1.25) -> dict[str, Tensor]:
    """CCML decomposition using the actual view count.

    Scale common evidence before powering to avoid overflow/underflow. All
    positive common vectors are algebraically identical to the pinned code.
    A zero common vector gives zero consistent evidence and legal alpha=1.
    rho>=1 covers the planned enhancement and no-enhancement ablation and
    has finite derivatives at zero (fractional powers below 1 do not).
    """
    _evidence(evidence, multiview=True)
    if not math.isfinite(rho) or rho < 1:
        raise ValueError("rho must be finite and >= 1")
    views = evidence.shape[1]
    common = evidence.amin(dim=1)
    residual = evidence - common.unsqueeze(1)
    complementary = residual.mean(dim=1)
    # Promoting the normalization also protects its backward pass: a float32
    # subnormal common value has an unrepresentable reciprocal in float32,
    # even when the complete derivative of the homogeneous transform is small.
    common_work = common.to(torch.float64)
    maximum = common_work.amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    powers = (common_work / scale).pow(rho)
    denominator = powers.sum(dim=-1, keepdim=True).clamp_min(1)
    consistent = (views * common_work.sum(dim=-1, keepdim=True) * (powers / denominator)).to(evidence.dtype)
    final = _finite_result(consistent + complementary, "fused evidence")
    return {"m": common, "e_con": consistent, "e_cmp": complementary, "e_final": final,
            **dirichlet_from_evidence(final)}


def predict_binary(probability: Tensor) -> Tensor:
    """MDD only when p_MDD > 0.5, with ties assigned to HC."""
    _finite_float(probability, "probability")
    if probability.ndim < 2 or probability.shape[-1] != 2:
        raise ValueError("binary probabilities must have a final dimension of size 2")
    if ((probability < 0) | (probability > 1)).any().item():
        raise ValueError("probabilities must lie in [0,1]")
    if not torch.allclose(probability.sum(-1), torch.ones_like(probability[..., 0]), atol=1e-6, rtol=1e-6):
        raise ValueError("class probabilities must sum to one")
    return (probability[..., 1] > 0.5).long()


def fuse_evidence(evidence: Tensor, method: str = "ccml", rho: float = 1.25) -> dict[str, Tensor | None]:
    """Fusion on fixed evidence, including TC-02/03 and modality subsets.

    Probability averaging intentionally has no fabricated alpha/e_final/u.
    """
    _evidence(evidence, multiview=True)
    if method == "ccml":
        return evidence_dc(evidence, rho)
    if method == "mean_probability":
        return {"p": dirichlet_from_evidence(evidence)["p"].mean(dim=1), "u": None,
                "alpha": None, "e_final": None}
    if method not in ("mean_evidence", "sum_evidence"):
        raise ValueError(f"unsupported fusion method: {method}")
    final = evidence.mean(dim=1) if method == "mean_evidence" else evidence.sum(dim=1)
    return {"e_final": final, **dirichlet_from_evidence(final)}


def fuse_predictions(logits: Tensor, method: str = "mean_probability", evidential: bool = True,
                     rho: float = 1.25) -> dict[str, Tensor | None]:
    """Convenience wrapper for raw head logits, preserving CE/EDL distinction."""
    _finite_float(logits, "logits")
    if logits.ndim != 3 or any(n == 0 for n in logits.shape) or logits.shape[-1] < 2:
        raise ValueError("logits must have nonempty [batch, view, class] shape")
    if evidential:
        return fuse_evidence(F.softplus(logits), method, rho)
    if method != "mean_probability":
        raise ValueError("ordinary CE heads support mean_probability fusion only")
    return {"p": logits.softmax(-1).mean(1), "u": None, "alpha": None, "e_final": None}


def annealing_coefficient(epoch: int | float, anneal_epochs: int = 50) -> float:
    """Training epochs are 1..60: coefficients 1/50 .. 1, then 1."""
    if not math.isfinite(epoch) or epoch < 0 or not math.isfinite(anneal_epochs) or anneal_epochs <= 0:
        raise ValueError("epoch must be nonnegative and anneal_epochs positive")
    return min(1.0, epoch / anneal_epochs)


def expected_cross_entropy(alpha: Tensor, one_hot: Tensor) -> Tensor:
    """Per-item expected CE for compatible [*, class] inputs."""
    return (one_hot * (torch.digamma(alpha.sum(-1, keepdim=True)) - torch.digamma(alpha))).sum(-1)


def kl_to_uniform(alpha: Tensor) -> Tensor:
    """Per-item KL(Dir(alpha) || Dir(1)); callers supply legal concentrations."""
    strength = alpha.sum(-1, keepdim=True)
    log_norm = torch.lgamma(strength).squeeze(-1) - torch.lgamma(alpha).sum(-1)
    log_uniform = torch.lgamma(alpha.new_tensor(float(alpha.shape[-1])))
    return log_norm - log_uniform + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength))).sum(-1)


def _loss_terms(evidence: Tensor, labels: Tensor, epoch: int, anneal_epochs: int) -> tuple[Tensor, Tensor]:
    one_hot = _targets(labels, evidence.shape[0], evidence.shape[-1], evidence.device).to(evidence.dtype)
    if evidence.ndim == 3:
        one_hot = one_hot.unsqueeze(1)
    alpha = evidence + 1
    data = expected_cross_entropy(alpha, one_hot)
    regularizer = annealing_coefficient(epoch, anneal_epochs) * kl_to_uniform(1 + (1 - one_hot) * evidence)
    return data, regularizer


def ce_heads_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """TC-01: unweighted CE summed over heads, mean over subjects."""
    _finite_float(logits, "logits")
    if logits.ndim != 3 or any(n == 0 for n in logits.shape) or logits.shape[-1] < 2:
        raise ValueError("logits must have nonempty [batch, view, class] shape")
    _targets(labels, logits.shape[0], logits.shape[-1], logits.device)
    targets = labels[:, None].expand(logits.shape[:2]).reshape(-1)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, reduction="none")
    return _finite_result(losses.reshape(logits.shape[:2]).sum(1).mean(), "CE loss")


def single_evidential_loss(evidence: Tensor, labels: Tensor, epoch: int, anneal_epochs: int = 50) -> Tensor:
    """TC-02: sum_v[D(alpha_v)+lambda KL(masked alpha_v)]."""
    _evidence(evidence, multiview=True)
    data, regularizer = _loss_terms(evidence, labels, epoch, anneal_epochs)
    return _finite_result((data + regularizer).sum(1).mean(), "single-view evidential loss")


def ccml_loss(evidence: Tensor, labels: Tensor, epoch: int, anneal_epochs: int = 50,
              rho: float = 1.25, delta: float = 1.0, gamma: float = 1.0) -> Tensor:
    """TC-04/05: single-view losses + D_con + delta R_con + gamma D_cmp.

    There is no complementary KL and no extra final fused CE term.
    gamma=0 removes complementary supervision, never inference evidence.
    """
    if not all(math.isfinite(x) and x >= 0 for x in (delta, gamma)):
        raise ValueError("delta and gamma must be finite and nonnegative")
    fused = evidence_dc(evidence, rho)
    views_data, views_regularizer = _loss_terms(evidence, labels, epoch, anneal_epochs)
    con_data, con_regularizer = _loss_terms(fused["e_con"], labels, epoch, anneal_epochs)
    one_hot = _targets(labels, evidence.shape[0], evidence.shape[-1], evidence.device).to(evidence.dtype)
    cmp_data = expected_cross_entropy(fused["e_cmp"] + 1, one_hot)
    loss = (views_data + views_regularizer).sum(1) + con_data + delta * con_regularizer + gamma * cmp_data
    return _finite_result(loss.mean(), "CCML loss")
