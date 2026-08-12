"""The verification ladder.

Five gates run in sequence; execution halts the instant one fails, saving
compute. Gates are plain objects satisfying the :class:`Gate` protocol, so
adding, removing, or reordering a gate is a change to the *list* handed to
:class:`VerificationLadder` — never a change to the runner, the high-level API,
or a future CLI. That is the whole point: no shotgun surgery when the ladder
evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from obelus.core.seams import ModelFactory, Scorer
from obelus.logging.tracker import NullTracker, Tracker

__all__ = ["GateResult", "LadderContext", "LadderReport", "Gate", "VerificationLadder"]


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single gate."""

    name: str
    passed: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LadderContext:
    """Everything the gates need, assembled once by the high-level API.

    Gates 1–3 verify the *baseline* architecture and slice sensitivity; Gates
    4–5 sweep the knockout ``variants`` for non-inferiority. ``variants`` maps a
    variant name to its knockout override set (``"baseline"`` -> ``{}``).
    """

    baseline_model: nn.Module
    model_factory: ModelFactory
    variants: dict[str, dict[str, Any]]
    scorer: Scorer
    slices: list[str]
    folds: int = 5
    input_shape: Optional[tuple[int, ...]] = None
    example_input: Optional[Callable[[], Any]] = None
    alpha: float = 0.05
    greater_is_better: bool = True
    effect_size: float = 0.05
    target_power: float = 0.8
    mutator_fraction: float = 0.30
    tracker: Tracker = field(default_factory=NullTracker)
    seed: int = 0

    def build_inputs(self) -> Optional[tuple]:
        """Return positional forward() args for gates 1-2, or ``None`` to skip.

        ``example_input`` wins when set: it returns a tensor (or tuple of them)
        and is the way to verify multi-input models — molecular nets taking
        graphs plus token ids, encoder/decoder pairs, anything whose forward
        signature is not one tensor. ``input_shape`` remains the shorthand for
        the common single-tensor case.
        """
        if self.example_input is not None:
            built = self.example_input()
            return built if isinstance(built, tuple) else (built,)
        if self.input_shape is not None:
            return (torch.randn(*self.input_shape),)
        return None


@runtime_checkable
class Gate(Protocol):
    """A single rung of the ladder."""

    name: str

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        """Execute against ``ctx``; ``prior`` holds already-run gate results."""
        ...


@dataclass(frozen=True)
class LadderReport:
    """Ordered gate results plus whether every gate ran and passed."""

    results: list[GateResult]
    passed: bool

    @property
    def halted_at(self) -> Optional[str]:
        """Name of the gate that failed, or ``None`` if the ladder passed."""
        for result in self.results:
            if not result.passed:
                return result.name
        return None

    def get(self, name: str) -> Optional[GateResult]:
        """Return the result for gate ``name`` if it ran, else ``None``."""
        for result in self.results:
            if result.name == name:
                return result
        return None


class VerificationLadder:
    """Runs gates in order, short-circuiting on the first failure."""

    def __init__(self, gates: list[Gate]) -> None:
        self.gates = gates

    def run(self, ctx: LadderContext) -> LadderReport:
        results: list[GateResult] = []
        with ctx.tracker.run("obelus_ladder"):
            for gate in self.gates:
                result = gate.run(ctx, results)
                results.append(result)
                ctx.tracker.log_metric(f"gate.{gate.name}.passed", float(result.passed))
                if not result.passed:
                    break
        passed = len(results) == len(self.gates) and all(r.passed for r in results)
        return LadderReport(results=results, passed=passed)
