"""Statistical power for the non-inferiority gate.

A non-inferiority gate is only trustworthy if it *could* have detected a real
degradation. Without power, ``p_degrade >= alpha`` conflates two very different
findings:

* **evidence of absence** — the slice is genuinely equivalent, or
* **absence of evidence** — the slice had no ability to tell.

The second silently RETAINS degraded variants, which is precisely the
"insensitive test battery / false confidence" failure obelus exists to prevent.
Two questions are answered here:

1. **Is the test structurally capable of rejecting at all?** A paired sign-flip
   permutation test over ``n`` folds has only ``2**n`` arrangements, so the
   smallest attainable one-tailed p-value is ``2**-n``. With ``alpha=0.05`` that
   makes 4 folds *impossible* to reject (``1/16 = 0.0625 > 0.05``) and 5 folds —
   the spec's default — only barely possible, requiring the single most extreme
   arrangement. This is exact arithmetic, not an estimate.

2. **How large a degradation could this slice detect?** Reported as the minimum
   detectable effect (MDE) at ``target_power``, plus the power to detect a
   user-declared ``margin`` (the smallest drop actually worth caring about).

Everything here is closed-form (noncentral *t*), so adding power costs
microseconds per cell rather than the thousands of resamples a Monte Carlo
estimate would need. :func:`simulate_power` is the slow, assumption-free
cross-check, kept for validating the approximation — not for report generation.

Post-hoc "observed power" (power computed at the *measured* effect) is
deliberately absent: it is a monotone restatement of the p-value and adds no
information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as st

__all__ = ["PowerReport", "min_achievable_p", "analyze_power", "simulate_power"]


@dataclass(frozen=True)
class PowerReport:
    """Power diagnostics for one baseline-vs-variant comparison on one slice."""

    n_folds: int
    alpha: float
    noise_sd: float
    min_achievable_p: float
    feasible: bool
    target_power: float
    mde: float
    margin: Optional[float] = None
    power_at_margin: Optional[float] = None

    @property
    def adequately_powered(self) -> bool:
        """Whether the test could detect the degradation you said you care about.

        Requires a ``margin``; without one the only defensible claim is
        structural feasibility, since adequacy is meaningless until you declare
        what size of drop matters.
        """
        if not self.feasible:
            return False
        if self.power_at_margin is None:
            return True  # nothing declared to be adequate *for*
        return self.power_at_margin >= self.target_power


def min_achievable_p(n_folds: int, n_resamples: int = 10_000) -> float:
    """Smallest one-tailed p-value the paired permutation test can produce.

    Exact enumeration of the ``2**n`` sign-flips gives a floor of ``2**-n``; when
    scipy falls back to random resampling the floor becomes
    ``1 / (n_resamples + 1)``.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    arrangements = 2**n_folds
    if arrangements <= n_resamples:
        return 1.0 / arrangements
    return 1.0 / (n_resamples + 1)


def analyze_power(
    baseline_scores: list[float],
    variant_scores: list[float],
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    margin: Optional[float] = None,
    greater_is_better: bool = True,
    n_resamples: int = 10_000,
) -> PowerReport:
    """Assess whether this slice's test could detect a real degradation.

    The paired differences supply the noise scale; the effect size is *declared*
    (``margin``) or *solved for* (``mde``) — never taken from the observed data,
    which would be post-hoc power.

    Uses the one-sided paired-*t* / noncentral-*t* approximation to the
    permutation test (the two agree closely, and the permutation test is the
    marginally more conservative of the pair at small ``n``), combined with the
    exact discreteness floor. ``margin`` is expressed in metric units as a
    positive degradation magnitude, e.g. ``0.03`` for "a 3-point F1 drop matters".
    """
    b = np.asarray(baseline_scores, dtype=float)
    v = np.asarray(variant_scores, dtype=float)
    if b.shape != v.shape:
        raise ValueError(
            f"paired scores must align: got {b.shape} baseline vs {v.shape} variant"
        )
    n = int(b.size)
    if n < 2:
        raise ValueError("need at least 2 paired folds to assess power")
    if margin is not None and margin < 0:
        raise ValueError("margin is a degradation magnitude and must be >= 0")

    sign = 1.0 if greater_is_better else -1.0
    diffs = sign * (v - b)
    noise_sd = float(diffs.std(ddof=1))

    floor = min_achievable_p(n, n_resamples)
    feasible = floor <= alpha

    df = n - 1
    t_crit = float(st.t.ppf(1.0 - alpha, df))

    def power_for(effect: float) -> float:
        if not feasible:
            return 0.0  # no arrangement can clear alpha, whatever the effect
        if effect <= 0:
            return alpha  # no true effect: rejection rate is the type-I rate
        if noise_sd == 0.0:
            return 1.0  # zero variance: any real shift is detected outright
        ncp = effect * math.sqrt(n) / noise_sd
        return float(st.nct.sf(t_crit, df, ncp))

    if not feasible:
        mde = math.inf
    elif noise_sd == 0.0:
        mde = 0.0
    else:
        # Standard one-sided paired-sample formula, inverted for the effect size.
        t_power = float(st.t.ppf(target_power, df))
        mde = (t_crit + t_power) * noise_sd / math.sqrt(n)

    return PowerReport(
        n_folds=n,
        alpha=alpha,
        noise_sd=noise_sd,
        min_achievable_p=floor,
        feasible=feasible,
        target_power=target_power,
        mde=mde,
        margin=margin,
        power_at_margin=None if margin is None else power_for(margin),
    )


def simulate_power(
    noise_sd: float,
    effect: float,
    n_folds: int,
    *,
    alpha: float = 0.05,
    n_simulations: int = 400,
    seed: int = 0,
) -> float:
    """Monte-Carlo power of the *actual* permutation test, for cross-checking.

    Slow by construction — one permutation test per simulation. Use it to
    validate :func:`analyze_power`, not inside a report.
    """
    from obelus.core.stats import evaluate_non_inferiority

    rng = np.random.default_rng(seed)
    zeros = np.zeros(n_folds)
    rejections = 0
    for _ in range(n_simulations):
        # The sign-flip test depends only on the paired differences, so
        # simulating those and testing them against zero is exactly equivalent.
        diffs = rng.normal(-effect, noise_sd, n_folds)
        result = evaluate_non_inferiority(zeros, diffs, alpha=alpha, seed=int(rng.integers(1 << 30)))
        rejections += not result.is_non_inferior
    return rejections / n_simulations
