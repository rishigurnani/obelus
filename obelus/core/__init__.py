"""Core engine: contracts, options, runner, mutator, stats, and the ladder."""

from __future__ import annotations

from obelus.core.contracts import check_invariants, check_shapes
from obelus.core.gates import default_gates
from obelus.core.ladder import (
    GateResult,
    LadderContext,
    LadderReport,
    VerificationLadder,
)
from obelus.core.mutator import ModelMutator
from obelus.core.options import COMPLEXIFY, SIMPLIFY, generate_moves
from obelus.core.policy import AblationPolicy, DecisionPolicy, InsertionPolicy
from obelus.core.power import PowerReport, analyze_power, min_achievable_p
from obelus.core.stats import classify_effect, evaluate_non_inferiority

__all__ = [
    "check_shapes",
    "check_invariants",
    "ModelMutator",
    "evaluate_non_inferiority",
    "classify_effect",
    "generate_moves",
    "SIMPLIFY",
    "COMPLEXIFY",
    "DecisionPolicy",
    "AblationPolicy",
    "InsertionPolicy",
    "analyze_power",
    "min_achievable_p",
    "PowerReport",
    "default_gates",
    "VerificationLadder",
    "LadderContext",
    "LadderReport",
    "GateResult",
]
