"""In-memory model mutator (Gate 3 sabotage)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from obelus.core.mutator import ModelMutator


def _linear_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))


def test_corruption_zeros_a_fraction_of_weights():
    model = _linear_model()
    gen = torch.Generator().manual_seed(7)
    corrupted = ModelMutator.create_corrupted_variant(model, 0.5, generator=gen)
    weight = dict(corrupted.named_parameters())["0.weight"]
    zeroed = (weight == 0).float().mean().item()
    assert 0.3 < zeroed < 0.7  # ~50% zeroed


def test_corruption_leaves_original_untouched():
    model = _linear_model()
    before = dict(model.named_parameters())["0.weight"].clone()
    ModelMutator.create_corrupted_variant(model, 0.5)
    after = dict(model.named_parameters())["0.weight"]
    assert torch.equal(before, after)


def test_corruption_is_deterministic_with_generator():
    model = _linear_model()
    a = ModelMutator.create_corrupted_variant(
        model, 0.3, generator=torch.Generator().manual_seed(1)
    )
    b = ModelMutator.create_corrupted_variant(
        model, 0.3, generator=torch.Generator().manual_seed(1)
    )
    wa = dict(a.named_parameters())["0.weight"]
    wb = dict(b.named_parameters())["0.weight"]
    assert torch.equal(wa, wb)


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        ModelMutator.create_corrupted_variant(_linear_model(), 1.5)


def test_bypass_residuals_replaces_named_children():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.residual = nn.Linear(4, 4)
            self.dense = nn.Linear(4, 4)

    out = ModelMutator.bypass_residuals(Net())
    assert isinstance(out.residual, nn.Identity)
    assert isinstance(out.dense, nn.Linear)
