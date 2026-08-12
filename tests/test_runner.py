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


class _ShiftInvariantGate(nn.Module):
    """Mirrors a real bug: softmax over dim=1 is shift-invariant, so the gate's
    scalar bias can never affect the output. Its true gradient is exactly 0, but
    float32 rounding usually yields ~1e-10 instead, and Adam's normalised step
    turns that into a full-size update — so a 'did the parameter move?' check
    reports this dead parameter as alive most of the time."""

    def __init__(self, dim=8):
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: (batch, branches, dim)
        weights = torch.softmax(self.gate(x), dim=1)
        return self.proj((weights * x).sum(1))


def test_structurally_dead_parameter_is_caught_deterministically():
    """The old did-it-move check passed ~5 runs in 6; this must fail every time."""
    for seed in range(6):
        torch.manual_seed(seed)
        with pytest.raises(PreflightError, match="gate.bias"):
            run_preflight(_ShiftInvariantGate(), (4, 3, 8), max_examples=1)


def test_partially_used_parameter_is_not_flagged():
    """An embedding whose unused rows get zero gradient is alive, not dead."""
    torch.manual_seed(0)
    inputs = (torch.randn(4, 8), torch.randint(0, 10, (4, 5)))

    class Partial(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4)
            self.embed = nn.Embedding(10, 4)  # most rows never indexed

        def forward(self, x, tokens):
            return self.lin(x) + self.embed(tokens).mean(1)

    run_preflight(Partial(), example_inputs=inputs, max_examples=1)
