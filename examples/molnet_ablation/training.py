"""Training and inference for the hybrid classifier.

Deliberately separate from both the model (a pure ``nn.Module``) and the ablation
wiring: obelus orchestrates *around* your training code without owning it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from examples.molnet_ablation.features import MolBatch


def fit(
    model: nn.Module,
    batch: MolBatch,
    y: np.ndarray,
    *,
    epochs: int = 14,
    lr: float = 3e-3,
    batch_size: int = 128,
    seed: int = 0,
) -> nn.Module:
    """Train ``model`` in place on a featurized batch; returns it for chaining."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    target = torch.tensor(np.asarray(y), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(batch)
    order = np.arange(n)
    model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(*batch.select(rows).as_args())
            loss = loss_fn(logits, target[rows])
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, batch: MolBatch) -> np.ndarray:
    """Return per-molecule probabilities of the positive (active) class."""
    model.eval()
    return torch.sigmoid(model(*batch.as_args())).cpu().numpy()
