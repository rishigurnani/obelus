"""Experiment-tracking seam.

The ladder logs through this narrow interface so nothing in the core imports
mlflow. ``NullTracker`` keeps offline tests dependency-free; ``MlflowTracker``
(in ``obelus.logging.mlflow_local``) is the real air-gapped SQLite adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

__all__ = ["Tracker", "NullTracker"]


@runtime_checkable
class Tracker(Protocol):
    """Minimal experiment logger."""

    @contextmanager
    def run(self, name: str) -> Iterator[None]:
        """Context manager delimiting one logical run (may be nested)."""
        ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...


class NullTracker:
    """No-op tracker used when tracking is disabled or unavailable."""

    @contextmanager
    def run(self, name: str) -> Iterator[None]:
        yield

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        pass
