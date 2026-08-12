"""Gate 1 — dynamic contracts & static invariants.

Two decorators users attach to custom PyTorch layers. They fire on every forward
pass (including when Hydra instantiates the module), so structural violations
surface at the boundary instead of as silent NaNs deep in a CV loop.

- ``check_shapes`` composes ``jaxtyped + beartype`` (so any jaxtyping
  annotations are enforced) *and* enforces a lightweight ``"in -> out"`` shape
  contract parsed from its spec string — the spec sketch ignored that string.
- ``check_invariants`` applies ``icontract`` pre/post-conditions for NaN/Inf
  inputs and output-norm explosion.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from beartype import beartype
from icontract import ensure, require
from jaxtyping import jaxtyped

__all__ = ["ShapeContractError", "check_shapes", "check_invariants"]


class ShapeContractError(TypeError):
    """Raised when a tensor's runtime shape violates a ``check_shapes`` spec."""


def _parse_side(text: str) -> list[list[str]]:
    """Parse one side of a shape spec into a list of per-tensor dim-token lists.

    ``"batch seq dim, batch dim"`` -> ``[["batch","seq","dim"], ["batch","dim"]]``
    """
    groups = [g.strip() for g in text.split(",") if g.strip()]
    return [g.split() for g in groups]


def _tensors(values) -> list[torch.Tensor]:
    return [v for v in values if isinstance(v, torch.Tensor)]


def _match(
    tensors: list[torch.Tensor],
    patterns: list[list[str]],
    bindings: dict[str, int],
    where: str,
) -> None:
    if len(tensors) < len(patterns):
        raise ShapeContractError(
            f"{where}: expected {len(patterns)} tensor(s) but found {len(tensors)}"
        )
    for idx, (tensor, tokens) in enumerate(zip(tensors, patterns)):
        if tensor.dim() != len(tokens):
            raise ShapeContractError(
                f"{where} tensor {idx}: expected rank {len(tokens)} "
                f"({' '.join(tokens)}) but got shape {tuple(tensor.shape)}"
            )
        for axis, (token, size) in enumerate(zip(tokens, tensor.shape)):
            size = int(size)
            if token.isdigit():
                if size != int(token):
                    raise ShapeContractError(
                        f"{where} tensor {idx} axis {axis}: expected {token}, got {size}"
                    )
            elif token in bindings:
                if bindings[token] != size:
                    raise ShapeContractError(
                        f"{where} tensor {idx} axis {axis}: dimension '{token}' "
                        f"bound to {bindings[token]} but got {size}"
                    )
            else:
                bindings[token] = size


def check_shapes(spec: str) -> Callable:
    """Enforce a ``"<inputs> -> <outputs>"`` shape contract on a forward method.

    Dimension names are bound left-to-right across inputs and must stay
    consistent on the outputs; integer tokens are exact sizes. Multiple tensors
    on a side are comma-separated. Runs *in addition* to ``jaxtyped(beartype)``.
    """
    if "->" not in spec:
        raise ValueError(f"shape spec must contain '->': {spec!r}")
    in_text, out_text = spec.split("->", 1)
    in_patterns = _parse_side(in_text)
    out_patterns = _parse_side(out_text)

    def decorator(fn: Callable) -> Callable:
        typed = jaxtyped(typechecker=beartype)(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bindings: dict[str, int] = {}
            inputs = _tensors(args) + _tensors(kwargs.values())
            _match(inputs, in_patterns, bindings, "input")
            result = typed(*args, **kwargs)
            outputs = result if isinstance(result, tuple) else (result,)
            _match(_tensors(outputs), out_patterns, bindings, "output")
            return result

        return wrapper

    return decorator


def check_invariants(
    no_nans: bool = True,
    no_infs: bool = True,
    max_norm_ratio: float = 10.0,
) -> Callable:
    """Apply pre/post-condition checks to the first tensor argument ``x``.

    Pre: ``x`` contains no NaNs / no Infs. Post: the output norm has not
    exploded beyond ``max_norm_ratio`` times the input norm. Non-floating
    tensors skip the NaN/Inf checks (they cannot hold those values).
    """

    def _no_nan(x) -> bool:
        if not no_nans or not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return True
        return not torch.isnan(x).any()

    def _no_inf(x) -> bool:
        if not no_infs or not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return True
        return not torch.isinf(x).any()

    def _norm_ok(result, x) -> bool:
        if not isinstance(result, torch.Tensor) or not isinstance(x, torch.Tensor):
            return True
        return float(result.detach().norm()) <= float(x.detach().norm()) * max_norm_ratio

    def decorator(fn: Callable) -> Callable:
        fn = require(_no_nan, description="Input contains NaNs")(fn)
        fn = require(_no_inf, description="Input contains Infs")(fn)
        fn = ensure(_norm_ok, description="Norm explosion detected")(fn)
        return fn

    return decorator
