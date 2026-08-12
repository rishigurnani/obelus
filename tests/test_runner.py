"""Gate 2 pre-flight verification."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from obelus.core.runner import PreflightError, run_preflight


def test_healthy_model_passes_preflight():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 8))
    run_preflight(model, (4, 8), max_examples=10)  # should not raise


def test_nan_producing_model_fails_fuzzing():
    class NanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8))

        def forward(self, x):
            return torch.log(x @ self.w)  # negative args -> NaN

    with pytest.raises(PreflightError):
        run_preflight(NanModel(), (4, 8), max_examples=50)


class _MultiInput(nn.Module):
    """Model whose forward takes a float tensor plus structural int/bool tensors."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.embed = nn.Embedding(10, 4)

    def forward(self, x, tokens, mask):
        return self.lin(x) + self.embed(tokens).mean(1) * mask.float().mean()


def test_multi_input_model_passes_via_example_inputs():
    torch.manual_seed(0)
    inputs = (
        torch.randn(4, 8),
        torch.randint(0, 10, (4, 5)),
        torch.ones(4, dtype=torch.bool),
    )
    run_preflight(_MultiInput(), example_inputs=inputs, max_examples=5)


def test_missing_both_input_specs_raises():
    with pytest.raises(ValueError):
        run_preflight(nn.Linear(8, 8))


def test_no_float_tensor_to_fuzz_raises():
    with pytest.raises(PreflightError):
        run_preflight(nn.Embedding(10, 4), example_inputs=(torch.randint(0, 10, (4, 5)),))


def test_dead_parameter_fails_gradient_check():
    class DeadParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(8, 8)
            self.dead = nn.Parameter(torch.randn(8, 8))  # never touched in forward

        def forward(self, x):
            return self.used(x)

    with pytest.raises(PreflightError):
        run_preflight(DeadParam(), (4, 8), max_examples=5)
