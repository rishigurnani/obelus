"""Gate 2 — pre-flight fuzzing & gradient integrity.

Fast checks run on an instantiated module *before* any expensive CV loop:

1. ``torchtest`` confirms parameters actually move under a gradient step
   (no dead sub-graphs, no detached branches).
2. ``hypothesis`` fuzzes the module with finite inputs and asserts it does not
   *introduce* NaN/Inf — i.e. it probes numerical stability, so the inputs are
   deliberately finite (feeding NaN in would trivially yield NaN out).
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from torchtest import assert_vars_change

__all__ = ["PreflightError", "run_preflight"]


class PreflightError(AssertionError):
    """Raised when a module fails a pre-flight gradient or stability check."""


def run_preflight(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str = "cpu",
    max_examples: int = 25,
    input_range: float = 10.0,
) -> None:
    """Run gradient-flow and numerical-stability checks; raise on failure.

    ``input_shape`` includes the batch dimension, e.g. ``(8, 16, 512)``.
    """
    # 1. Gradient update verification: every trainable parameter must change.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dummy = torch.randn(*input_shape, device=device)
    try:
        assert_vars_change(
            model=model,
            # torchtest calls loss_fn(output, target); we only need a scalar
            # that depends on every output element, so the target is ignored.
            loss_fn=lambda out, _target: out.float().pow(2).mean(),
            optim=optimizer,
            batch=[dummy, torch.zeros(1, device=device)],
            device=device,
        )
    except Exception as exc:  # torchtest raises its own exception types
        raise PreflightError(f"gradient integrity check failed: {exc}") from exc

    # 2. Property fuzzing for numerical stability on *finite* inputs.
    model.eval()

    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(
        x=arrays(
            dtype=np.float32,
            shape=input_shape,
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
            out = model(torch.from_numpy(x).to(device))
        if torch.isnan(out).any():
            raise PreflightError("pre-flight fuzz failed: NaN produced from finite input")
        if torch.isinf(out).any():
            raise PreflightError("pre-flight fuzz failed: Inf produced from finite input")

    _fuzz()
