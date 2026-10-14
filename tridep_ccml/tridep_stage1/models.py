"""Paired low-capacity heads for frozen CBraMod window/channel/time tokens.

All four conditions use exactly the same learned modules. Only fixed pooling
of the original token axes changes. Equal parameter counts do not imply equal
effective function capacity after an input axis has been removed by pooling.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CONDITIONS = ("mean", "spatial", "temporal", "spatiotemporal")
TOKEN_SHAPE = (60, 19, 5, 200)


class CBraModTokenHead(nn.Module):
    """Frozen [batch, 60, 19, 5, 200] tokens -> 128D subject -> CE logits.

    ``adjacency[i, j]`` is the fixed contribution from channel j to channel i.
    The caller determines electrode order and graph construction before fitting.
    The matrix must already have unit row sums and a positive diagonal.
    """

    def __init__(self, condition: str, adjacency: torch.Tensor):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown token-head condition: {condition}")
        graph = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
        if graph.shape != (19, 19):
            raise ValueError("Electrode adjacency must be 19 x 19")
        if not torch.isfinite(graph).all() or torch.any(graph < 0):
            raise ValueError("Electrode adjacency must be finite and nonnegative")
        if not torch.allclose(graph.sum(1), torch.ones_like(graph.sum(1)), atol=1e-6, rtol=0):
            raise ValueError("Electrode adjacency must have unit row sums")
        if torch.any(torch.diag(graph) <= 0):
            raise ValueError("Electrode adjacency requires a positive self-loop")
        self.condition = condition
        self.use_spatial = condition in ("spatial", "spatiotemporal")
        self.use_temporal = condition in ("temporal", "spatiotemporal")
        self.register_buffer("adjacency", graph)
        self.token_projection = nn.Linear(200, 32)
        self.spatial_projection = nn.Linear(32, 32)
        self.temporal_conv = nn.Conv1d(32, 32, kernel_size=3, groups=32)
        self.window_projection = nn.Linear(32, 128)
        self.classifier = nn.Linear(128, 2)

    def window_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return [batch, 60, 128] before subject-level window averaging."""
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != TOKEN_SHAPE or tokens.shape[0] == 0:
            raise ValueError("Expected nonempty frozen [batch, 60, 19, 5, 200] tokens")
        if not torch.is_floating_point(tokens):
            raise ValueError("CBraMod tokens must be floating point")
        # This module trains the new head only, including when a caller passes
        # a tensor attached to an encoder computation graph.
        x = tokens.detach()
        if not self.use_spatial:
            x = x.mean(dim=2, keepdim=True).expand(-1, -1, 19, -1, -1)
        if not self.use_temporal:
            x = x.mean(dim=3, keepdim=True).expand(-1, -1, -1, 5, -1)
        x = F.relu(self.token_projection(x))
        # The same fixed graph is applied in every condition. With the spatial
        # axis pooled, all channel vectors are equal and propagation preserves
        # them, up to floating-point summation error.
        x = torch.einsum("ij,bwjtd->bwitd", self.adjacency, x)
        x = F.relu(self.spatial_projection(x))
        batch = x.shape[0]
        x = x.permute(0, 1, 2, 4, 3).reshape(batch * 60 * 19, 32, 5)
        x = F.relu(self.temporal_conv(F.pad(x, (1, 1), mode="replicate")))
        x = x.reshape(batch, 60, 19, 32, 5).permute(0, 1, 2, 4, 3)
        pooled = x.mean(dim=(2, 3))
        return F.relu(self.window_projection(pooled))

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return [batch, 128], averaging all 60 window representations."""
        return self.window_features(tokens).mean(dim=1)

    def forward(self, tokens: torch.Tensor, return_features: bool = False):
        representation = self.encode(tokens)
        logits = self.classifier(representation)
        return (logits, representation) if return_features else logits

    def architecture(self) -> dict:
        """Return the actual configuration and parameter counts for registration."""
        return {
            "condition": self.condition,
            "input_shape_per_subject": list(TOKEN_SHAPE),
            "disabled_axes_pooled_before_token_projection": [
                axis for axis, active in (("channel", self.use_spatial), ("time", self.use_temporal)) if not active
            ],
            "token_projection": "Linear(200,32), ReLU",
            "spatial_operator": "fixed row-stochastic 19-channel adjacency with self-loops",
            "spatial_projection": "Linear(32,32), ReLU",
            "temporal_operator": "depthwise Conv1d(32,32,kernel=3), replicate padding, ReLU",
            "window_projection": "mean(channel,time), Linear(32,128), ReLU",
            "subject_aggregation": "mean of all 60 window representations",
            "classifier": "Linear(128,2), raw CE logits",
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "parameter_counts_by_module": {
                name: sum(p.numel() for p in module.parameters())
                for name, module in self.named_children()
            },
            "frozen_input": True,
            "capacity_interpretation": "Identical learned skeleton and parameter counts; pooling changes input information and effective function capacity.",
        }
