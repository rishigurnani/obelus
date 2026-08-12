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


def test_seed_makes_pvalue_reproducible():
    baseline = [0.80, 0.75, 0.90, 0.60, 0.85]
    variant = [0.70, 0.72, 0.66, 0.58, 0.71]
    a = evaluate_non_inferiority(baseline, variant, seed=42)
    b = evaluate_non_inferiority(baseline, variant, seed=42)
    assert a.p_degrade == b.p_degrade
