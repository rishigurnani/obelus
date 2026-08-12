"""End-to-end ablation through the high-level API (offline seams)."""

from __future__ import annotations

import pandas as pd

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
    assert {"variant", "slice", "p_degrade", "is_non_inferior", "decision"} <= set(df.columns)


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
        seed=0,
    )
    assert (out / "summary.csv").exists()
