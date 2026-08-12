"""Air-gapped MLflow tracker.

Local SQLite tracking with ecosystem telemetry suppressed, so a run is 100%
offline. Import stays lazy: ``mlflow`` is an optional extra and the core never
depends on it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["MlflowTracker", "init_local_tracker"]

# Telemetry flags for common libraries, set before mlflow touches the network.
_TELEMETRY_ENV = {
    "LIGHTNING_DISABLE_TELEMETRY": "1",
    "RAY_USAGE_STATS_ENABLED": "0",
    "HF_HUB_OFFLINE": "1",
    "MLFLOW_DISABLE_INSECURE_REQUEST_WARNING": "true",
}


def _suppress_telemetry() -> None:
    for key, value in _TELEMETRY_ENV.items():
        os.environ.setdefault(key, value)


class MlflowTracker:
    """:class:`~obelus.logging.tracker.Tracker` backed by a local MLflow store."""

    def __init__(
        self,
        db_path: str = "sqlite:///obelus_experiments.db",
        experiment: str = "obelus_non_inferiority_ablations",
    ) -> None:
        import mlflow  # optional dependency, imported lazily

        _suppress_telemetry()
        self._mlflow = mlflow
        mlflow.set_tracking_uri(db_path)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str) -> Iterator[None]:
        # nested=True lets per-variant runs live under an outer ablation run.
        active = self._mlflow.active_run() is not None
        with self._mlflow.start_run(run_name=name, nested=active):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._mlflow.log_metric(key, value, step=step)


def init_local_tracker(
    db_path: str = "sqlite:///obelus_experiments.db",
    experiment: str = "obelus_non_inferiority_ablations",
) -> MlflowTracker:
    """Disable ecosystem telemetry and return a local SQLite MLflow tracker."""
    return MlflowTracker(db_path=db_path, experiment=experiment)
