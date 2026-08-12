"""Power analysis for the non-inferiority gate."""

from __future__ import annotations

import math

import pytest

from obelus.core.power import analyze_power, min_achievable_p, simulate_power


def test_discreteness_floor_is_exact():
    assert min_achievable_p(5) == pytest.approx(1 / 32)
    assert min_achievable_p(10) == pytest.approx(1 / 1024)


def test_floor_falls_back_to_resampling_limit_for_many_folds():
    # 2**20 arrangements > n_resamples, so scipy randomizes: floor = 1/(N+1).
    assert min_achievable_p(20, n_resamples=999) == pytest.approx(1 / 1000)


def test_four_folds_cannot_reject_at_five_percent():
    """1/16 = 0.0625 > 0.05: no arrangement clears alpha, so power is exactly 0."""
    report = analyze_power([0.5] * 4, [0.1, 0.2, 0.15, 0.12], alpha=0.05, margin=0.5)
    assert report.min_achievable_p == pytest.approx(0.0625)
    assert report.feasible is False
    assert report.power_at_margin == 0.0
    assert report.adequately_powered is False
    assert math.isinf(report.mde)


def test_five_folds_are_feasible_but_only_just():
    report = analyze_power([0.5] * 5, [0.4, 0.42, 0.38, 0.41, 0.39], alpha=0.05)
    assert report.feasible is True
    assert report.min_achievable_p == pytest.approx(0.03125)


def test_noisy_slice_has_a_larger_mde_than_a_stable_one():
    stable = analyze_power([0.70, 0.71, 0.69, 0.70, 0.71], [0.70, 0.71, 0.69, 0.70, 0.71])
    noisy = analyze_power([0.70, 0.71, 0.69, 0.70, 0.71], [0.40, 0.95, 0.55, 0.85, 0.60])
    assert noisy.mde > stable.mde


def test_more_folds_shrink_the_mde():
    few = analyze_power([0.5, 0.6, 0.4, 0.55, 0.45], [0.52, 0.57, 0.43, 0.53, 0.47])
    many = analyze_power(
        [0.5, 0.6, 0.4, 0.55, 0.45] * 4, [0.52, 0.57, 0.43, 0.53, 0.47] * 4
    )
    assert many.mde < few.mde


def test_adequacy_requires_a_declared_margin_to_be_meaningful():
    scores = [0.5, 0.62, 0.44, 0.58, 0.46]
    variant = [0.51, 0.60, 0.45, 0.57, 0.47]
    tiny_margin = analyze_power(scores, variant, margin=0.001)  # far below the MDE
    big_margin = analyze_power(scores, variant, margin=0.5)  # far above it
    assert tiny_margin.adequately_powered is False
    assert big_margin.adequately_powered is True
    assert big_margin.power_at_margin > tiny_margin.power_at_margin


def test_power_rises_with_effect_size():
    b = [0.5, 0.62, 0.44, 0.58, 0.46]
    v = [0.51, 0.60, 0.45, 0.57, 0.47]
    powers = [analyze_power(b, v, margin=m).power_at_margin for m in (0.01, 0.05, 0.2)]
    assert powers[0] < powers[1] < powers[2]


def test_mde_is_the_effect_reaching_target_power():
    b = [0.5, 0.62, 0.44, 0.58, 0.46]
    v = [0.51, 0.60, 0.45, 0.57, 0.47]
    report = analyze_power(b, v, target_power=0.8)
    at_mde = analyze_power(b, v, target_power=0.8, margin=report.mde)
    assert at_mde.power_at_margin == pytest.approx(0.8, abs=0.02)


def test_metric_direction_is_respected():
    # Identical spread either way: the noise scale should not depend on direction.
    b = [0.30, 0.35, 0.25, 0.32, 0.28]
    v = [0.31, 0.34, 0.26, 0.33, 0.27]
    assert analyze_power(b, v).mde == pytest.approx(
        analyze_power(b, v, greater_is_better=False).mde
    )


def test_analytic_power_approximates_the_real_permutation_test():
    """Cross-check the closed form against the actual test it approximates."""
    noise_sd, effect, folds = 0.05, 0.08, 10
    b = [0.5] * folds
    # Construct paired scores with the target noise scale to feed analyze_power.
    import numpy as np

    rng = np.random.default_rng(0)
    diffs = rng.normal(0.0, noise_sd, folds)
    diffs = diffs / diffs.std(ddof=1) * noise_sd  # pin the SD exactly
    v = list(np.asarray(b) + diffs)

    analytic = analyze_power(b, v, alpha=0.05, margin=effect).power_at_margin
    empirical = simulate_power(noise_sd, effect, folds, alpha=0.05, n_simulations=300)
    assert analytic == pytest.approx(empirical, abs=0.15)


def test_rejects_negative_margin_and_too_few_folds():
    with pytest.raises(ValueError):
        analyze_power([0.5, 0.6], [0.4, 0.5], margin=-0.1)
    with pytest.raises(ValueError):
        analyze_power([0.5], [0.4])
