"""End-to-end ablation through the high-level API (offline seams)."""

from __future__ import annotations

import pandas as pd
import torch.nn as nn

from conftest import DIM, make_factory, make_scorer

from obelus import AutoAblate, run_ablation
from obelus.core.ladder import LadderReport


def _run(**overrides):
    kwargs = dict(
        scorer=make_scorer(),
        model_factory=make_factory(),
        variants={"weak_variant": {"scale": 0.2}, "strong_variant": {"scale": 1.05}},
        slices=["val_full", "slice_long_seq", "slice_noisy"],
        input_shape=(4, DIM),
        cv_folds=5,
        p_alpha=0.05,
        effect_size=0.05,
        seed=0,
    )
    kwargs.update(overrides)
    return run_ablation(**kwargs)


def test_full_ladder_passes_and_all_gates_run():
    result = _run()
    assert result.passed is True
    names = [r.name for r in result.report.results]
    assert names == ["contracts", "preflight", "fire_drill", "cv_sweep", "non_inferiority"]


def test_decisions_split_variants_correctly():
    result = _run()
    decisions = result.report.get("non_inferiority").data["decisions"]
    assert decisions["weak_variant"] == "REJECTED"
    assert decisions["strong_variant"] == "RETAINED"
    assert set(result.retained) == {"strong_variant"}


def test_summary_dataframe_shape_and_columns():
    result = _run()
    df = result.summary
    assert isinstance(df, pd.DataFrame)
    # 2 variants x 3 slices = 6 rows
    assert len(df) == 6
    assert {"variant", "slice", "label", "p_backward", "p_forward", "decision"} <= set(df.columns)


def _paired_run(baseline_scores, variant_scores, **kwargs):
    """Drive the ladder with scripted per-fold scores for one slice."""
    scores = {"baseline": baseline_scores, "v": variant_scores}

    def factory(overrides):
        model = nn.Linear(2, 2)
        model.tag = overrides.get("tag", "baseline")
        return model

    def scorer(model, slice_name, fold):
        return scores[model.tag][fold]

    return run_ablation(
        scorer=scorer,
        model_factory=factory,
        variants={"v": {"tag": "v"}},
        slices=["s"],
        cv_folds=len(baseline_scores),
        sanity_mutation=False,
        seed=0,
        effect_size=kwargs.pop("effect_size", 0.05),
        **kwargs,
    )


def test_underpowered_null_result_is_flagged_inconclusive():
    # Large per-fold differences that cancel out: no significant effect, and the
    # noise swamps a 0.03 margin. (Power depends on the paired *differences*,
    # not on how spread out the raw scores are.)
    base = [0.50, 0.80, 0.35, 0.75, 0.40, 0.70, 0.45, 0.65, 0.55, 0.60]
    var = [0.60, 0.71, 0.43, 0.64, 0.47, 0.64, 0.57, 0.57, 0.60, 0.52]
    row = _paired_run(base, var, effect_size=0.03).summary.iloc[0]
    assert row["label"] == "ON_PAR"
    assert bool(row["adequately_powered"]) is False
    assert bool(row["inconclusive"]) is True
    assert row["mde"] > 0.03  # cannot resolve the declared margin


def test_well_powered_null_result_is_not_flagged():
    # Tight, near-identical scores: not significant, but able to see 0.03.
    base = [0.700, 0.702, 0.699, 0.701, 0.703, 0.698, 0.700, 0.702, 0.699, 0.701]
    var = [0.701, 0.701, 0.700, 0.700, 0.702, 0.699, 0.701, 0.701, 0.700, 0.700]
    row = _paired_run(base, var, effect_size=0.03).summary.iloc[0]
    assert row["label"] == "ON_PAR"
    assert bool(row["adequately_powered"]) is True
    assert bool(row["inconclusive"]) is False
    assert row["mde"] < 0.03


def test_gate_summary_counts_underpowered_cells():
    base = [0.50, 0.80, 0.35, 0.75, 0.40, 0.70, 0.45, 0.65, 0.55, 0.60]
    var = [0.60, 0.71, 0.43, 0.64, 0.47, 0.64, 0.57, 0.57, 0.60, 0.52]
    report = _paired_run(base, var, effect_size=0.03).report
    gate = report.get("non_inferiority")
    assert gate.data["n_inconclusive"] == 1
    assert "underpowered" in gate.summary


def test_disabling_sanity_mutation_skips_fire_drill():
    result = _run(sanity_mutation=False)
    names = [r.name for r in result.report.results]
    assert "fire_drill" not in names
    assert result.passed is True


def test_autoablate_returns_dataframe_with_report_in_attrs():
    df = AutoAblate(
        scorer=make_scorer(),
        model_factory=make_factory(),
        variants={"weak_variant": {"scale": 0.2}},
        slices=["val_full", "slice_noisy"],
        input_shape=(4, DIM),
        effect_size=0.05,
        seed=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.attrs["ladder_report"], LadderReport)


def test_autoablate_writes_summary_csv(tmp_path):
    out = tmp_path / "reports"
    AutoAblate(
        scorer=make_scorer(),
        model_factory=make_factory(),
        variants={"weak_variant": {"scale": 0.2}},
        slices=["val_full"],
        input_shape=(4, DIM),
        output_dir=str(out),
        effect_size=0.05,
        seed=0,
    )
    assert (out / "summary.csv").exists()


def test_on_par_everywhere_is_rejected():
    """A variant must earn its place: no forward leap anywhere -> REJECTED."""
    base = [0.700, 0.702, 0.699, 0.701, 0.703, 0.698, 0.700, 0.702, 0.699, 0.701]
    var = [0.701, 0.701, 0.700, 0.700, 0.702, 0.699, 0.701, 0.701, 0.700, 0.700]
    result = _paired_run(base, var, effect_size=0.03)
    assert result.summary.iloc[0]["label"] == "ON_PAR"
    assert result.report.get("non_inferiority").data["decisions"]["v"] == "REJECTED"


def test_forward_leap_with_no_regression_is_retained():
    base = [0.50, 0.60, 0.70, 0.80, 0.90, 0.55, 0.65, 0.75, 0.85, 0.45]
    var = [0.70, 0.81, 0.89, 1.00, 1.10, 0.76, 0.84, 0.96, 1.04, 0.66]
    result = _paired_run(base, var, effect_size=0.10)
    assert result.summary.iloc[0]["label"] == "FORWARD"
    assert result.report.get("non_inferiority").data["decisions"]["v"] == "RETAINED"
