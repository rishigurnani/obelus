"""Non-inferiority permutation test."""

from __future__ import annotations

import pytest

from obelus.core.stats import evaluate_non_inferiority


def test_clear_degradation_is_flagged():
    baseline = [0.80, 0.82, 0.79, 0.81, 0.83]
    variant = [0.60, 0.62, 0.58, 0.61, 0.59]
    res = evaluate_non_inferiority(baseline, variant, alpha=0.05, seed=0)
    assert res.p_degrade < 0.05
    assert res.is_non_inferior is False


def test_non_inferior_variant_passes():
    baseline = [0.80, 0.82, 0.79, 0.81, 0.83]
    variant = [0.81, 0.83, 0.80, 0.82, 0.84]  # slightly better
    res = evaluate_non_inferiority(baseline, variant, alpha=0.05, seed=0)
    assert res.p_degrade >= 0.05
    assert res.is_non_inferior is True


def test_lower_is_better_flips_direction():
    # As a loss metric, the "variant" here is better (lower) -> non-inferior.
    baseline = [0.50, 0.52, 0.49, 0.51, 0.53]
    variant = [0.30, 0.32, 0.29, 0.31, 0.33]
    worse = evaluate_non_inferiority(baseline, variant, greater_is_better=True, seed=0)
    better = evaluate_non_inferiority(baseline, variant, greater_is_better=False, seed=0)
    assert worse.is_non_inferior is False  # treated as accuracy: a big drop
    assert better.is_non_inferior is True  # treated as loss: an improvement


def test_constant_uniform_drop_shortcut():
    # Identical paired differences -> degenerate permutation; decided directly.
    baseline = [1.0, 1.0, 1.0, 1.0, 1.0]
    variant = [0.5, 0.5, 0.5, 0.5, 0.5]
    res = evaluate_non_inferiority(baseline, variant, seed=0)
    assert res.p_degrade == 0.0
    assert res.is_non_inferior is False


def test_identical_scores_are_non_inferior():
    scores = [0.7, 0.71, 0.69, 0.72, 0.70]
    res = evaluate_non_inferiority(scores, list(scores), seed=0)
    assert res.p_degrade == 1.0
    assert res.is_non_inferior is True


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        evaluate_non_inferiority([1.0, 2.0], [1.0], seed=0)


def test_too_few_folds_raise():
    with pytest.raises(ValueError):
        evaluate_non_inferiority([1.0], [0.5], seed=0)


def test_tiny_mean_gain_is_not_called_an_improvement():
    """A higher mean is not an improvement unless the test says so."""
    baseline = [0.52, 0.48, 0.60, 0.44, 0.55]
    variant = [0.53, 0.44, 0.63, 0.42, 0.58]  # mean up a hair, direction inconsistent
    res = evaluate_non_inferiority(baseline, variant, alpha=0.05, seed=0)
    assert res.variant_mean > res.baseline_mean  # the misleading part
    assert res.is_non_inferior is True  # not significantly worse ...
    assert res.is_improved is False  # ... but not significantly better either
    assert res.p_improve >= 0.05


def test_consistent_gain_is_a_significant_improvement():
    baseline = [0.50, 0.60, 0.70, 0.80, 0.90]
    variant = [0.52, 0.61, 0.72, 0.81, 0.92]  # every fold improves
    res = evaluate_non_inferiority(baseline, variant, alpha=0.05, seed=0)
    assert res.is_improved is True
    assert res.p_improve < 0.05
    assert res.is_non_inferior is True  # an improvement is also non-inferior


def test_degradation_is_never_reported_as_improvement():
    baseline = [0.80, 0.82, 0.79, 0.81, 0.83]
    variant = [0.60, 0.62, 0.58, 0.61, 0.59]
    res = evaluate_non_inferiority(baseline, variant, alpha=0.05, seed=0)
    assert res.is_non_inferior is False
    assert res.is_improved is False
    assert res.p_improve > res.p_degrade


def test_identical_scores_are_neither_degraded_nor_improved():
    scores = [0.7, 0.71, 0.69, 0.72, 0.70]
    res = evaluate_non_inferiority(scores, list(scores), seed=0)
    assert res.is_non_inferior is True
    assert res.is_improved is False
    assert res.p_improve == 1.0


def test_uniform_gain_shortcut_reports_improvement():
    # Constant paired differences take the non-permutation shortcut path.
    res = evaluate_non_inferiority([1.0] * 5, [1.5] * 5, seed=0)
    assert res.p_degrade == 1.0
    assert res.p_improve == 0.0
    assert res.is_improved is True


def test_improvement_respects_metric_direction():
    # As a loss metric, lower variant scores are the improvement.
    baseline = [0.50, 0.60, 0.70, 0.80, 0.90]
    variant = [0.48, 0.59, 0.68, 0.79, 0.88]
    as_loss = evaluate_non_inferiority(baseline, variant, greater_is_better=False, seed=0)
    assert as_loss.is_improved is True


def test_reported_means_match_the_inputs():
    baseline = [0.80, 0.90, 1.00]
    variant = [0.20, 0.30, 0.40]
    res = evaluate_non_inferiority(baseline, variant, seed=0)
    assert res.baseline_mean == pytest.approx(0.90)
    assert res.variant_mean == pytest.approx(0.30)


def test_acceptance_boundary_is_inclusive():
    """The spec's rule is ``p_degrade >= alpha``, so p == alpha must PASS."""
    baseline = [0.50, 0.60, 0.70, 0.80, 0.90]
    variant = [0.40, 0.55, 0.60, 0.75, 0.80]
    # 5 paired folds -> exact enumeration; the most extreme arrangement is 1/32.
    res = evaluate_non_inferiority(baseline, variant, alpha=0.03125, seed=0)
    assert res.p_degrade == pytest.approx(0.03125)
    assert res.is_non_inferior is True  # strict '>' would wrongly reject here


def test_test_is_paired_not_independent():
    """A small but perfectly consistent per-fold drop must register.

    Each fold degrades, yet the two score distributions overlap almost entirely,
    so an *unpaired* test sees nothing (p ~ 0.42). Pairing is what makes this
    detectable, and the spec requires a paired permutation test.
    """
    baseline = [0.50, 0.60, 0.70, 0.80, 0.90]
    variant = [0.48, 0.59, 0.68, 0.79, 0.88]
    res = evaluate_non_inferiority(baseline, variant, alpha=0.05, seed=0)
    assert res.p_degrade == pytest.approx(0.03125)
    assert res.is_non_inferior is False


def test_seed_makes_pvalue_reproducible():
    baseline = [0.80, 0.75, 0.90, 0.60, 0.85]
    variant = [0.70, 0.72, 0.66, 0.58, 0.71]
    a = evaluate_non_inferiority(baseline, variant, seed=42)
    b = evaluate_non_inferiority(baseline, variant, seed=42)
    assert a.p_degrade == b.p_degrade
