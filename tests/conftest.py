"""Shared offline testbed.

A tiny model whose *score* is a deterministic function of its weight magnitude,
so corruption (weight zeroing) and config scaling both move the score in a
predictable direction — letting the whole ladder run without data or a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from obelus.core.contracts import check_shapes

DIM = 8


class TinyModel(nn.Module):
    """A single linear layer carrying a shape contract (exercises Gate 1)."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    @check_shapes("batch dim -> batch dim")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def make_factory(dim: int = DIM):
    """Build a model whose weight magnitude scales with an override ``scale``."""

    def factory(overrides) -> nn.Module:
        scale = float(overrides.get("scale", 1.0))
        gen = torch.Generator().manual_seed(1234)
        model = TinyModel(dim)
        with torch.no_grad():
            for param in model.parameters():
                base = torch.rand(param.shape, generator=gen) * 0.5 + 0.1
                param.copy_(base * scale)  # strictly positive, monotonic in scale
        return model

    return factory


def make_scorer():
    """Score == total weight magnitude, with small deterministic per-fold noise."""

    def scorer(model: nn.Module, slice_name: str, fold: int) -> float:
        with torch.no_grad():
            mag = sum(float(p.abs().sum()) for p in model.parameters())
        noise = 0.05 * ((fold % 5) - 2) / 2.0  # varied so paired diffs aren't constant
        return mag * (1.0 + noise)

    return scorer
