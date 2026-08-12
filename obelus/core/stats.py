"""Non-inferiority statistical engine.

A one-tailed *paired* permutation test asking a single question: is the variant
significantly *worse* than the baseline? Everything downstream (Gate 3's fire
drill and Gate 5's non-inferiority decision) is expressed in terms of this one
primitive, so there is exactly one place where the statistics live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats

__all__ = ["DegradationTest", "evaluate_non_inferiority"]


@dataclass(frozen=True)
class DegradationTest:
    """Outcome of a single baseline-vs-variant comparison on one slice."""

    p_degrade: float
    is_non_inferior: bool
    baseline_mean: float
    variant_mean: float


def evaluate_non_inferiority(
    baseline_scores: list[float],
    variant_scores: list[float],
    alpha: float = 0.05,
    *,
    greater_is_better: bool = True,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> DegradationTest:
    """Test whether ``variant`` is significantly worse than ``baseline``.

    Scores are *paired* per fold, so ``baseline_scores[k]`` and
    ``variant_scores[k]`` must come from the same fold ``k``.

    Null hypothesis H0: variant is degraded (variant < baseline).
    Alternative H1: variant is non-inferior (variant >= baseline).

    Returns the probability that the observed (or worse) degradation arose by
    chance, ``p_degrade``. The variant is deemed **non-inferior** when a
    degradation is *not* statistically significant, i.e. ``p_degrade >= alpha``.

    ``greater_is_better`` flips the comparison so the same engine handles
    lower-is-better metrics (loss) without a second code path.
    """
    b = np.asarray(baseline_scores, dtype=float)
    v = np.asarray(variant_scores, dtype=float)
    if b.shape != v.shape:
        raise ValueError(
            f"paired scores must align: got {b.shape} baseline vs {v.shape} variant"
        )
    if b.size < 2:
        raise ValueError("need at least 2 paired folds to run a permutation test")

    # Orient so that "worse" always means a smaller number: for loss-style
    # metrics we negate, collapsing both directions onto one lower-tail test.
    sign = 1.0 if greater_is_better else -1.0
    b_o, v_o = sign * b, sign * v

    def statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
        # variant - baseline; a significantly negative value => degradation.
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    diffs = v_o - b_o
    if np.allclose(diffs, diffs[0]):
        # Constant paired differences give the permutation test no variation to
        # work with (it would emit a degenerate p-value). Decide directly: a
        # uniform drop is a certain degradation; anything else is not.
        p_degrade = 0.0 if diffs[0] < 0 else 1.0
    else:
        res = stats.permutation_test(
            (v_o, b_o),
            statistic=statistic,
            permutation_type="samples",  # paired sign-flip permutations
            vectorized=True,
            n_resamples=n_resamples,
            alternative="less",  # tests variant < baseline (degradation)
            rng=seed,
        )
        p_degrade = float(res.pvalue)

    return DegradationTest(
        p_degrade=p_degrade,
        is_non_inferior=p_degrade >= alpha,
        baseline_mean=float(b.mean()),
        variant_mean=float(v.mean()),
    )
