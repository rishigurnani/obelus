"""Knockout generation from a Hydra-style config."""

from __future__ import annotations

from omegaconf import OmegaConf

from obelus.core.discovery import generate_knockouts


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "_target_": "src.models.Transformer",
                "attn": {"_target_": "src.layers.CustomAttention", "dim": 512},
                "norm": {"_target_": "torch.nn.LayerNorm"},  # library module: not knocked out
                "dropout": {"active": True, "p": 0.1},
            }
        }
    )


def test_baseline_is_always_present():
    knockouts = generate_knockouts(_cfg())
    assert knockouts["baseline"] == {}


def test_custom_src_module_is_knocked_to_identity():
    knockouts = generate_knockouts(_cfg())
    assert knockouts["no_attn"] == {"model.attn._target_": "torch.nn.Identity"}


def test_active_toggle_is_disabled():
    knockouts = generate_knockouts(_cfg())
    assert knockouts["no_dropout"] == {"model.dropout.active": False}


def test_library_module_is_not_ablated():
    knockouts = generate_knockouts(_cfg())
    assert "no_norm" not in knockouts
