"""Real-edge adapters that bind the injectable seams to Hydra/PyTorch.

These are the only places that touch Hydra instantiation. Tests bypass them
with plain callables, keeping the ladder runnable offline.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

__all__ = ["HydraModelFactory"]


class HydraModelFactory:
    """Builds a model by applying knockout ``overrides`` to a base Hydra config.

    Satisfies :class:`obelus.core.seams.ModelFactory`. Override keys are dotted
    paths rooted at the config (e.g. ``"model.attn._target_"``), as produced by
    :func:`obelus.core.options.generate_moves`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def __call__(self, overrides: Mapping[str, Any]) -> nn.Module:
        from hydra.utils import instantiate  # lazy: only the real path needs hydra

        cfg = OmegaConf.create(OmegaConf.to_container(self._cfg, resolve=False))
        for dotted, value in overrides.items():
            # merge=False so a dict override *replaces* the node rather than
            # merging into it — an option swap must not leave the previous
            # option's keys (or the _options_/_current_ scaffolding) behind.
            OmegaConf.update(cfg, dotted, value, merge=False, force_add=True)
        return instantiate(cfg.model)
