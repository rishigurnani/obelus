"""Paired cross-validation evaluation — the mechanism shared by Gate 3 and Gate 5.

Both gates ask the same underlying question: score two models across every
``(slice, fold)`` cell, then run the permutation test per slice. They differ
*only* in what verdict they want (Gate 3 expects degradation to be detected;
Gate 5 expects non-inferiority). That shared question lives here exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from obelus.core.seams import Scorer
from obelus.core.stats import DegradationTest, evaluate_non_inferiority

__all__ = ["SlicePair", "SliceVerdict", "evaluate_pair", "compare_across_slices"]


@dataclass(frozen=True)
class SlicePair:
    """Paired baseline/variant scores for one slice, aligned by fold index."""

    slice_name: str
    baseline: list[float]
    variant: list[float]


@dataclass(frozen=True)
class SliceVerdict:
    """A slice's paired scores together with its degradation test result."""

    slice_name: str
    baseline: list[float]
    variant: list[float]
    test: DegradationTest


def evaluate_pair(
    baseline: nn.Module,
    variant: nn.Module,
    slices: list[str],
    folds: int,
    scorer: Scorer,
) -> list[SlicePair]:
    """Score ``baseline`` and ``variant`` on every ``(slice, fold)`` cell."""
    pairs: list[SlicePair] = []
    for slice_name in slices:
        b = [float(scorer(baseline, slice_name, k)) for k in range(folds)]
        v = [float(scorer(variant, slice_name, k)) for k in range(folds)]
        pairs.append(SlicePair(slice_name, b, v))
    return pairs


def compare_across_slices(
    baseline: nn.Module,
    variant: nn.Module,
    slices: list[str],
    folds: int,
    scorer: Scorer,
    *,
    alpha: float = 0.05,
    greater_is_better: bool = True,
    seed: int | None = None,
) -> list[SliceVerdict]:
    """Evaluate a pair across slices and run the degradation test on each."""
    verdicts: list[SliceVerdict] = []
    for pair in evaluate_pair(baseline, variant, slices, folds, scorer):
        test = evaluate_non_inferiority(
            pair.baseline,
            pair.variant,
            alpha=alpha,
            greater_is_better=greater_is_better,
            seed=seed,
        )
        verdicts.append(
            SliceVerdict(pair.slice_name, pair.baseline, pair.variant, test)
        )
    return verdicts
