"""Real-edge Hydra adapter: instantiation and override application."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import OmegaConf

from obelus.adapters import HydraModelFactory


def _cfg():
    return OmegaConf.create(
        {"model": {"_target_": "torch.nn.Linear", "in_features": 8, "out_features": 8}}
    )


def test_factory_instantiates_baseline():
    model = HydraModelFactory(_cfg())({})
    assert isinstance(model, nn.Linear)
    assert model.in_features == 8


def test_factory_applies_knockout_override():
    # A knockout swaps the target to Identity; leftover kwargs are ignored by it.
    model = HydraModelFactory(_cfg())({"model._target_": "torch.nn.Identity"})
    assert isinstance(model, nn.Identity)


def test_factory_does_not_mutate_base_cfg():
    cfg = _cfg()
    HydraModelFactory(cfg)({"model._target_": "torch.nn.Identity"})
    assert cfg.model._target_ == "torch.nn.Linear"
