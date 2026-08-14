"""Ordered `_options_` config: one declaration, both directions."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from obelus.core.options import COMPLEXIFY, SIMPLIFY, find_option_nodes, generate_moves


def _cfg():
    """Mid-stack architecture: one component present, two absent."""
    return OmegaConf.create(
        {
            "model": {
                "_target_": "src.models.Net",
                "encoder": {
                    "_options_": [
                        {"_target_": "torch.nn.Identity"},
                        {"_target_": "src.layers.Encoder", "dim": 8},
                    ],
                    "_current_": 1,  # present -> can be simplified away
                },
                "adapter": {
                    "_options_": [
                        {"_target_": "torch.nn.Identity"},
                        {"_target_": "src.layers.Adapter"},
                    ],
                    "_current_": 0,  # absent -> can be inserted
                },
            }
        }
    )


def test_finds_every_decision_node():
    nodes = find_option_nodes(_cfg())
    assert set(nodes) == {"model.encoder", "model.adapter"}
    assert nodes["model.encoder"].current == 1


def test_simplify_offers_only_lower_ranked_options():
    moves = generate_moves(_cfg(), SIMPLIFY)
    assert set(moves) == {"baseline", "no_encoder"}  # adapter is already simplest


def test_complexify_offers_only_higher_ranked_options():
    moves = generate_moves(_cfg(), COMPLEXIFY)
    assert set(moves) == {"baseline", "add_adapter"}  # encoder is already fullest


def test_baseline_pins_every_node_to_its_incumbent():
    baseline = generate_moves(_cfg(), SIMPLIFY)["baseline"]
    assert baseline["model.encoder"]["_target_"] == "src.layers.Encoder"
    assert baseline["model.adapter"]["_target_"] == "torch.nn.Identity"


def test_variant_differs_from_baseline_in_exactly_one_node():
    moves = generate_moves(_cfg(), COMPLEXIFY)
    base, variant = moves["baseline"], moves["add_adapter"]
    differing = [k for k in base if base[k] != variant[k]]
    assert differing == ["model.adapter"]


def test_three_ranks_yield_both_directions_from_one_node():
    cfg = OmegaConf.create(
        {
            "model": {
                "enc": {
                    "_options_": [
                        {"_target_": "torch.nn.Identity"},
                        {"_target_": "src.layers.Small"},
                        {"_target_": "src.layers.Big"},
                    ],
                    "_current_": 1,
                }
            }
        }
    )
    assert set(generate_moves(cfg, SIMPLIFY)) == {"baseline", "no_enc"}
    assert set(generate_moves(cfg, COMPLEXIFY)) == {"baseline", "add_enc"}


@pytest.mark.parametrize("direction", [SIMPLIFY, COMPLEXIFY])
def test_a_bare_target_config_is_rejected_in_both_directions(direction):
    """One standard: undeclared alternatives are an error, not a silent fallback.

    A bare ``_target_`` used to be auto-knocked-out by a prefix heuristic. That
    guess is gone — a knockout is now just the rank-0 option of a declared node,
    so a config with nothing declared has no moves to derive.
    """
    cfg = OmegaConf.create({"model": {"attn": {"_target_": "src.layers.Attn"}}})
    with pytest.raises(ValueError, match="no _options_ nodes found"):
        generate_moves(cfg, direction)


@pytest.mark.parametrize(
    "node", [{"_options_": [{"a": 1}], "_current_": 0}, {"_options_": [{"a": 1}, {"b": 2}], "_current_": 5}]
)
def test_malformed_nodes_are_rejected(node):
    with pytest.raises(ValueError):
        find_option_nodes(OmegaConf.create({"model": {"x": node}}))
