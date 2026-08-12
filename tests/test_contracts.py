"""Gate 1 building blocks: check_shapes and check_invariants."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from obelus.core.contracts import (
    ShapeContractError,
    check_invariants,
    check_shapes,
)


class CustomAttention(nn.Module):
    """The spec's example layer, carrying both contracts."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    @check_shapes("batch seq dim -> batch seq dim")
    @check_invariants(no_nans=True, no_infs=True, max_norm_ratio=10.0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_check_shapes_accepts_valid_shape():
    layer = CustomAttention(dim=16)
    out = layer(torch.randn(2, 4, 16))
    assert out.shape == (2, 4, 16)


def test_check_shapes_rejects_wrong_rank():
    layer = CustomAttention(dim=16)
    with pytest.raises(ShapeContractError):
        layer(torch.randn(4, 16))  # rank 2, contract wants rank 3


def test_check_shapes_binds_dimensions_consistently():
    @check_shapes("batch dim -> batch dim")
    def identity(x):
        return x[:, :-1]  # breaks the 'dim' binding on output

    with pytest.raises(ShapeContractError):
        identity(torch.randn(3, 5))


def test_check_shapes_enforces_integer_literals():
    @check_shapes("batch 8 -> batch 8")
    def keep(x):
        return x

    keep(torch.randn(3, 8))
    with pytest.raises(ShapeContractError):
        keep(torch.randn(3, 7))


def test_check_invariants_rejects_nan_input():
    layer = CustomAttention(dim=4)
    bad = torch.full((1, 2, 4), float("nan"))
    with pytest.raises(Exception):  # icontract ViolationError
        layer(bad)


def test_check_invariants_rejects_norm_explosion():
    @check_invariants(no_nans=True, no_infs=True, max_norm_ratio=2.0)
    def amplify(x):
        return x * 100.0

    with pytest.raises(Exception):
        amplify(torch.ones(4))


def test_check_invariants_allows_bounded_output():
    @check_invariants(max_norm_ratio=2.0)
    def shrink(x):
        return x * 0.5

    out = shrink(torch.ones(4))
    assert torch.allclose(out, torch.full((4,), 0.5))
