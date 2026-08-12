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


def test_custom_prefixes_are_configurable():
    cfg = OmegaConf.create(
        {"model": {"enc": {"_target_": "myproj.layers.Encoder"}}}
    )
    assert "no_enc" not in generate_knockouts(cfg)  # default prefix is "src."
    knockouts = generate_knockouts(cfg, custom_prefixes=("myproj.",))
    assert knockouts["no_enc"] == {"model.enc._target_": "torch.nn.Identity"}
