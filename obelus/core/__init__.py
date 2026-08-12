"""Core engine: contracts, discovery, runner, mutator, stats, and the ladder."""

from __future__ import annotations

from obelus.core.contracts import check_invariants, check_shapes
from obelus.core.discovery import generate_knockouts
from obelus.core.gates import default_gates
from obelus.core.ladder import (
    GateResult,
    LadderContext,
    LadderReport,
    VerificationLadder,
)
from obelus.core.mutator import ModelMutator
from obelus.core.stats import evaluate_non_inferiority

__all__ = [
    "check_shapes",
    "check_invariants",
    "generate_knockouts",
    "ModelMutator",
    "evaluate_non_inferiority",
    "default_gates",
    "VerificationLadder",
    "LadderContext",
    "LadderReport",
    "GateResult",
]
