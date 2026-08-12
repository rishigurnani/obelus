"""Topology introspection — Gate 4's knockout matrix generator.

Scans a Hydra model config for custom (``src.``) target modules and ``active``
toggles, emitting one knockout override set per ablatable module. Each override
set is a flat ``{dotted.path: value}`` mapping consumed by a ``ModelFactory``.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

__all__ = ["generate_knockouts"]


def generate_knockouts(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    """Return knockout override sets keyed by variant name.

    Always includes ``"baseline"`` (an empty override set). For every custom
    ``src.*`` module a ``no_<name>`` variant swaps its ``_target_`` to
    ``torch.nn.Identity``; for every ``active`` toggle a ``no_<name>`` variant
    sets it ``False``.
    """
    knockouts: dict[str, dict[str, Any]] = {"baseline": {}}
    container = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(container, dict):
        return knockouts

    def _scan(node: dict, path: str = "model") -> None:
        for key, value in node.items():
            curr = f"{path}.{key}"
            if isinstance(value, dict):
                target = value.get("_target_")
                if target is not None and "src." in str(target):
                    knockouts[f"no_{key}"] = {f"{curr}._target_": "torch.nn.Identity"}
                elif "active" in value:
                    knockouts[f"no_{key}"] = {f"{curr}.active": False}
                _scan(value, curr)

    _scan(container)
    return knockouts
