"""In-memory model mutator — the sabotage half of Gate 3's fire drill.

Programmatically corrupts a *copy* of a model so the ladder can confirm the
user's evaluation slices are sensitive enough to notice a known-bad model
before spending GPU hours on real variants.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

__all__ = ["ModelMutator"]


class ModelMutator:
    """In-memory semantic mutator for verifying benchmark slice sensitivity."""

    @staticmethod
    def create_corrupted_variant(
        model: nn.Module,
        fraction_zero: float = 0.30,
        *,
        generator: torch.Generator | None = None,
    ) -> nn.Module:
        """Return a deep copy of ``model`` with ~``fraction_zero`` of the weights
        in each multi-dimensional parameter zeroed out.

        Pass a seeded ``generator`` for reproducible corruption in tests.
        """
        if not 0.0 <= fraction_zero <= 1.0:
            raise ValueError(f"fraction_zero must be in [0, 1], got {fraction_zero}")
        corrupted = copy.deepcopy(model)
        with torch.no_grad():
            for param in corrupted.parameters():
                if param.requires_grad and param.dim() > 1:
                    keep = torch.rand(
                        param.shape, generator=generator, device=param.device
                    ) > fraction_zero
                    param.mul_(keep.to(param.dtype))
        return corrupted

    @staticmethod
    def bypass_residuals(model: nn.Module) -> nn.Module:
        """Return a deep copy with residual/skip child modules replaced by
        ``nn.Identity`` (severing skip connections)."""
        corrupted = copy.deepcopy(model)
        for name, _child in list(corrupted.named_children()):
            lname = name.lower()
            if "residual" in lname or "skip" in lname:
                setattr(corrupted, name, nn.Identity())
        return corrupted
