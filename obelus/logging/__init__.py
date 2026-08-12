"""Experiment tracking seam and the air-gapped MLflow adapter."""

from __future__ import annotations

from obelus.logging.tracker import NullTracker, Tracker

__all__ = ["Tracker", "NullTracker", "init_local_tracker", "MlflowTracker"]


def __getattr__(name: str):
    # Lazy re-export so importing obelus.logging never pulls in mlflow.
    if name in ("init_local_tracker", "MlflowTracker"):
        from obelus.logging import mlflow_local

        return getattr(mlflow_local, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
