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

__all__ = [
    "DegradationTest",
    "EffectClass",
    "evaluate_non_inferiority",
    "classify_effect",
]

FORWARD, ON_PAR, BACKWARD = "FORWARD", "ON_PAR", "BACKWARD"


@dataclass(frozen=True)
class DegradationTest:
    """Outcome of a single baseline-vs-variant comparison on one slice.

    Both tails of the same paired permutation test are reported, because "not
    significantly worse" and "significantly better" are different claims and a
    mean gap alone justifies neither.
    """

    p_degrade: float
    is_non_inferior: bool
    baseline_mean: float
    variant_mean: float
    p_improve: float
    is_improved: bool


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

    The opposite tail of the same test is reported as ``p_improve``, so an
    *improvement* can be stated with the same rigour: ``is_improved`` requires
    ``p_improve < alpha``, never merely a higher mean. Non-inferiority and
    improvement are distinct claims — most non-inferior variants are simply
    indistinguishable from the baseline, not better than it.

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

    def one_tailed(alternative: str) -> float:
        res = stats.permutation_test(
            (v_o, b_o),
            statistic=statistic,
            permutation_type="samples",  # paired sign-flip permutations
            vectorized=True,
            n_resamples=n_resamples,
            alternative=alternative,
            rng=seed,
        )
        return float(res.pvalue)

    diffs = v_o - b_o
    if np.allclose(diffs, diffs[0]):
        # Constant paired differences give the permutation test no variation to
        # work with (it would emit a degenerate p-value). Decide directly: a
        # uniform shift is certain in its direction; no shift is neither.
        p_degrade = 0.0 if diffs[0] < 0 else 1.0
        p_improve = 0.0 if diffs[0] > 0 else 1.0
    else:
        p_degrade = one_tailed("less")  # variant < baseline (degradation)
        p_improve = one_tailed("greater")  # variant > baseline (improvement)

    return DegradationTest(
        p_degrade=p_degrade,
        is_non_inferior=p_degrade >= alpha,
        baseline_mean=float(b.mean()),
        variant_mean=float(v.mean()),
        p_improve=p_improve,
        is_improved=p_improve < alpha,
    )


@dataclass(frozen=True)
class EffectClass:
    """Where a variant lands relative to a *declared* meaningful effect size."""

    label: str  # FORWARD | ON_PAR | BACKWARD
    p_forward: float
    p_backward: float
    baseline_mean: float
    variant_mean: float


def classify_effect(
    baseline_scores: list[float],
    variant_scores: list[float],
    effect_size: float,
    alpha: float = 0.05,
    *,
    greater_is_better: bool = True,
    seed: int | None = None,
) -> EffectClass:
    """Classify a variant against a meaningful ``effect_size`` delta.

    A difference only counts once it is both **large enough to matter** and
    **statistically significant**, which is what the bare non-inferiority test
    cannot express: it treats any drop as a regression, however trivial.

    No new statistics are involved — testing against a margin is the same paired
    permutation test with the baseline shifted by the margin, so this runs
    :func:`evaluate_non_inferiority` against ``b - delta`` and ``b + delta``:

    * ``BACKWARD`` — significantly worse than ``baseline - delta``
    * ``FORWARD``  — significantly better than ``baseline + delta``
    * ``ON_PAR``   — neither leap is established
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0; declare the delta that matters")
    b = np.asarray(baseline_scores, dtype=float)
    shift = (1.0 if greater_is_better else -1.0) * effect_size
    kwargs = dict(alpha=alpha, greater_is_better=greater_is_better, seed=seed)
    back = evaluate_non_inferiority(b - shift, variant_scores, **kwargs)
    fwd = evaluate_non_inferiority(b + shift, variant_scores, **kwargs)
    label = (
        BACKWARD if back.p_degrade < alpha
        else FORWARD if fwd.p_improve < alpha
        else ON_PAR
    )
    return EffectClass(label, fwd.p_improve, back.p_degrade, float(b.mean()), back.variant_mean)
