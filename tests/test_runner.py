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
