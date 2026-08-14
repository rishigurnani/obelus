"""Ordered architectural options — the variant generator for both directions.

A decision node lists its alternatives **ordered simple → complex** and names the
one currently in force::

    model:
      graph_encoder:
        _options_:
          - {_target_: torch.nn.Identity}          # rank 0
          - {_target_: src.layers.GraphEncoder}    # rank 1
        _current_: 1

The ordering *is* the user's declaration of what "simpler" means, so obelus never
has to guess. From one config it derives both directions:

* **simplify** (ablation)   — every option ranked *below* ``_current_``
* **complexify** (insertion) — every option ranked *above* ``_current_``

This is the only variant form obelus supports. It subsumes the knockout
shorthand it replaced — swapping a ``_target_`` to ``torch.nn.Identity`` is just
a two-option node with Identity at rank 0, and an on/off toggle is a two-option
node over the flag — while also expressing insertions, which a shorthand that
can only *subtract* never could. Because alternatives are declared rather than
discovered, there is no prefix heuristic guessing which modules are yours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf


__all__ = ["OptionNode", "SIMPLIFY", "COMPLEXIFY", "find_option_nodes", "generate_moves"]

SIMPLIFY, COMPLEXIFY = "simplify", "complexify"
_OPTIONS, _CURRENT = "_options_", "_current_"


@dataclass(frozen=True)
class OptionNode:
    """One decision point: its ranked alternatives and the incumbent's rank."""

    path: str
    options: list[dict]
    current: int

    def option_name(self, rank: int) -> str:
        target = str(self.options[rank].get("_target_", ""))
        leaf = self.path.rsplit(".", 1)[-1]
        if target.endswith("Identity"):
            return f"no_{leaf}"
        if rank > self.current:
            return f"add_{leaf}"
        return f"simpler_{leaf}"


def find_option_nodes(cfg: DictConfig) -> dict[str, OptionNode]:
    """Locate every ``_options_`` decision node, keyed by dotted config path."""
    container = OmegaConf.to_container(cfg, resolve=True)
    nodes: dict[str, OptionNode] = {}

    def _scan(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if _OPTIONS in node:
            options = node[_OPTIONS]
            if not isinstance(options, list) or len(options) < 2:
                raise ValueError(f"{path}: {_OPTIONS} must list at least two alternatives")
            current = node.get(_CURRENT)
            if not isinstance(current, int) or not 0 <= current < len(options):
                raise ValueError(
                    f"{path}: {_CURRENT} must be an index into {_OPTIONS} "
                    f"(0..{len(options) - 1}), got {current!r}"
                )
            nodes[path] = OptionNode(path, options, current)
            return  # options are leaves; do not descend into the alternatives
        for key, value in node.items():
            _scan(value, f"{path}.{key}" if path else str(key))

    _scan(container, "")
    return nodes


def generate_moves(cfg: DictConfig, direction: str = SIMPLIFY) -> dict[str, dict[str, Any]]:
    """Return one override set per available move in ``direction``.

    Every override set also pins each *other* decision node to its incumbent, so
    a variant differs from the baseline in exactly one decision and the
    ``_options_``/``_current_`` scaffolding never reaches ``instantiate``.
    """
    if direction not in (SIMPLIFY, COMPLEXIFY):
        raise ValueError(f"direction must be {SIMPLIFY!r} or {COMPLEXIFY!r}, got {direction!r}")

    nodes = find_option_nodes(cfg)
    if not nodes:
        raise ValueError(
            "no _options_ nodes found: declare each decision point's alternatives "
            "ordered simple -> complex, with _current_ naming the incumbent — "
            "obelus derives moves from that, it does not guess"
        )

    incumbent = {path: node.options[node.current] for path, node in nodes.items()}
    moves: dict[str, dict[str, Any]] = {"baseline": dict(incumbent)}
    for path, node in nodes.items():
        ranks = (
            range(node.current) if direction == SIMPLIFY
            else range(node.current + 1, len(node.options))
        )
        for rank in ranks:
            overrides = dict(incumbent)
            overrides[path] = node.options[rank]
            name = node.option_name(rank)
            if name in moves:  # several alternatives on one node
                name = f"{name}_r{rank}"
            moves[name] = overrides
    return moves
