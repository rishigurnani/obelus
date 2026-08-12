"""Real air-gapped MLflow tracking over local SQLite."""

from __future__ import annotations

import os

import mlflow

from conftest import DIM, make_factory, make_scorer

from obelus import AutoAblate
from obelus.logging import init_local_tracker


def test_init_local_tracker_suppresses_telemetry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    init_local_tracker("sqlite:///telemetry_probe.db")
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0"
    assert os.environ["LIGHTNING_DISABLE_TELEMETRY"] == "1"


def test_ablation_logs_runs_and_gate_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep all mlflow files inside the tmp dir
    db = "sqlite:///obelus_experiments.db"

    AutoAblate(
        scorer=make_scorer(),
        model_factory=make_factory(),
        variants={"weak_variant": {"scale": 0.2}, "strong_variant": {"scale": 1.05}},
        slices=["val_full", "slice_noisy"],
        input_shape=(4, DIM),
        db_path=db,
        effect_size=0.05,
        seed=0,
    )

    mlflow.set_tracking_uri(db)
    experiment = mlflow.get_experiment_by_name("obelus_non_inferiority_ablations")
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    # outer ladder run + one nested run per variant
    assert len(runs) >= 3

    metric_cols = [c for c in runs.columns if c.startswith("metrics.gate.")]
    assert any("non_inferiority" in c for c in metric_cols)
