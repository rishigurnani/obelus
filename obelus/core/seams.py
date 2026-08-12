"""Injectable seams.

The heavy, environment-specific concerns — how a model is built from a config
and how a model is scored on a data slice — are expressed as narrow protocols so
the verification ladder can run fully offline in tests with deterministic
stand-ins, while real Hydra/PyTorch/data adapters plug in at the edges.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import torch.nn as nn

__all__ = ["ModelFactory", "Scorer"]


@runtime_checkable
class ModelFactory(Protocol):
    """Builds a model from knockout ``overrides`` applied to a base config.

    ``overrides`` is a flat mapping of dotted config paths to replacement values
    (see :func:`obelus.core.discovery.generate_knockouts`). An empty mapping
    yields the baseline model.
    """

    def __call__(self, overrides: Mapping[str, Any]) -> nn.Module: ...


@runtime_checkable
class Scorer(Protocol):
    """Scores a model on one ``(slice, fold)`` cell.

    Higher is better by default (see ``greater_is_better`` on the stats/ladder
    APIs for loss-style metrics). Implementations own all data loading and
    evaluation — obelus never touches the user's training scripts.
    """

    def __call__(self, model: nn.Module, slice_name: str, fold: int) -> float: ...
