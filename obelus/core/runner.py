"""Gate 2 — pre-flight fuzzing & gradient integrity.

Fast checks run on an instantiated module *before* any expensive CV loop:

1. ``torchtest`` confirms parameters actually move under a gradient step
   (no dead sub-graphs, no detached branches).
2. ``hypothesis`` fuzzes the module with finite inputs and asserts it does not
   *introduce* NaN/Inf — i.e. it probes numerical stability, so the inputs are
   deliberately finite (feeding NaN in would trivially yield NaN out).

Multi-input models are supported: pass ``example_inputs``. The first
floating-point tensor is the one varied and fuzzed; the remaining arguments
(token ids, adjacency, masks) are held fixed, since perturbing a structural
integer tensor with random floats would test nothing meaningful.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from torchtest import assert_vars_change

__all__ = ["PreflightError", "run_preflight"]


class PreflightError(AssertionError):
    """Raised when a module fails a pre-flight gradient or stability check."""


class _VaryOne(nn.Module):
    """Expose a multi-input model as a single-tensor model.

    Lets ``torchtest`` and ``hypothesis`` — both of which assume ``model(x)`` —
    drive a model whose forward takes several tensors, by substituting the
    varied argument and closing over the rest. Parameters are inherited from the
    wrapped model, so gradient checks still see the real ones.
    """

    def __init__(self, model: nn.Module, inputs: Sequence, index: int) -> None:
        super().__init__()
        self.inner = model
        self._inputs = list(inputs)
        self._index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = list(self._inputs)
        args[self._index] = x
        return self.inner(*args)


def _dead_parameters(
    model: nn.Module,
    inputs: Sequence,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
) -> list[str]:
    """Names of trainable parameters that receive no usable gradient.

    Checks the **gradient** rather than whether an optimizer moved the parameter.
    That distinction matters: Adam normalises its step to
    ``lr * g / (|g| + eps)``, so a gradient of 1e-10 — pure float32 rounding
    noise around a structurally-zero gradient — still produces a full-size
    update. A "did the parameter change?" test therefore reports a dead
    parameter as alive whenever the noise happens not to round to exactly zero,
    which is most of the time. Reading the gradient is invariant to the
    optimizer and to that noise.

    Runs in eval mode so dropout cannot zero a path by chance and make a live
    parameter look dead.
    """
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        model(*inputs).float().pow(2).mean().backward()
        grads = {
            name: (0.0 if p.grad is None else float(p.grad.abs().max()))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        if not grads:
            return []
        # Scale-relative floor: a gradient far below the model's largest is
        # noise, not signal, whatever the absolute units happen to be.
        threshold = max(abs_tol, rel_tol * max(grads.values()))
        return [name for name, g in grads.items() if g <= threshold]
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def _varied_index(inputs: Sequence) -> int:
    """Index of the first floating-point tensor — the one worth perturbing."""
    for i, value in enumerate(inputs):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return i
    raise PreflightError("no floating-point tensor among the example inputs to fuzz")


def run_preflight(
    model: torch.nn.Module,
    input_shape: Optional[tuple[int, ...]] = None,
    *,
    example_inputs: Optional[Sequence] = None,
    device: str = "cpu",
    max_examples: int = 25,
    input_range: float = 10.0,
) -> None:
    """Run gradient-flow and numerical-stability checks; raise on failure.

    Provide either ``input_shape`` (single-tensor model; includes the batch
    dimension, e.g. ``(8, 16, 512)``) or ``example_inputs`` (a sequence of
    concrete forward arguments, for multi-input models).
    """
    if example_inputs is None:
        if input_shape is None:
            raise ValueError("provide either input_shape or example_inputs")
        example_inputs = (torch.randn(*input_shape, device=device),)

    index = _varied_index(example_inputs)
    probe = _VaryOne(model, example_inputs, index)
    varied = example_inputs[index]

    # 1a. Structural gradient check, run before anything mutates the weights.
    # Deterministic, and it catches the dead parameters torchtest's
    # did-it-move test misses (see _dead_parameters).
    dead = _dead_parameters(model, example_inputs)
    if dead:
        raise PreflightError(
            "gradient integrity check failed: no usable gradient reaches "
            + ", ".join(dead)
        )

    # 1b. Gradient update verification: every trainable parameter must change.
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    try:
        assert_vars_change(
            model=probe,
            # torchtest calls loss_fn(output, target); we only need a scalar
            # that depends on every output element, so the target is ignored.
            loss_fn=lambda out, _target: out.float().pow(2).mean(),
            optim=optimizer,
            batch=[varied, torch.zeros(1, device=device)],
            device=device,
        )
    except Exception as exc:  # torchtest raises its own exception types
        raise PreflightError(f"gradient integrity check failed: {exc}") from exc

    # 2. Property fuzzing for numerical stability on *finite* inputs.
    probe.eval()

    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(
        x=arrays(
            dtype=np.float32,
            shape=tuple(varied.shape),
            elements=st.floats(
                min_value=-input_range,
                max_value=input_range,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )
    def _fuzz(x: np.ndarray) -> None:
        with torch.no_grad():
            out = probe(torch.from_numpy(x).to(device))
        if torch.isnan(out).any():
            raise PreflightError("pre-flight fuzz failed: NaN produced from finite input")
        if torch.isinf(out).any():
            raise PreflightError("pre-flight fuzz failed: Inf produced from finite input")

    _fuzz()
