"""Composable architecture, assembled by Hydra from ``configs/architecture.yaml``.

Every submodule here is an independent **architectural decision**. Because each
is a ``_target_`` in the config, ``obelus.core.discovery.generate_knockouts``
finds it automatically and emits a variant that replaces it with
``torch.nn.Identity`` — no ablation code, and no hand-written variant list.

The knockout convention: a branch swapped to ``nn.Identity`` counts as *absent*
(see :func:`_present`), so the fusion layer simply receives one fewer embedding.
That is what makes a config-level ``_target_`` swap a real architectural change
rather than a crash.

Layout::

    descriptors ──► DescriptorBranch   (2D computed features)  ┐
    atom graph  ──► GraphEncoder       (learned, GNN)          ├─► Fusion ─► head
    SMILES ids  ──► SequenceEncoder    (learned, Transformer)  ┘
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from examples.molnet_ablation.features import (
    ATOM_FEAT_DIM,
    MAX_LEN,
    N_DESCRIPTORS,
    VOCAB_SIZE,
)
from obelus.core.contracts import check_invariants, check_shapes


def _present(module: Optional[nn.Module]) -> bool:
    """A branch knocked out to ``nn.Identity`` is treated as absent."""
    return module is not None and not isinstance(module, nn.Identity)


def _toggled_on(flag: Any) -> bool:
    """Read an ``active:`` toggle, which discovery flips to ``False``."""
    if flag is None:
        return False
    if isinstance(flag, bool):
        return flag
    return bool(flag.get("active", True))


class DescriptorBranch(nn.Module):
    """Hand-crafted 2D RDKit descriptors -> embedding."""

    def __init__(self, in_dim: int = N_DESCRIPTORS, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.net(descriptors)


class GraphEncoder(nn.Module):
    """Learned features: normalized graph convolutions over the atom graph.

    ``residual`` is an ``active:`` toggle, so obelus generates a ``no_residual``
    variant that severs the skip connections without touching this code.
    """

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 2,
        in_dim: int = ATOM_FEAT_DIM,
        residual: Any = None,
    ):
        super().__init__()
        self.residual = _toggled_on(residual)
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(layers)
        )

    def forward(
        self, atom_feats: torch.Tensor, adjacency: torch.Tensor, atom_mask: torch.Tensor
    ) -> torch.Tensor:
        h = atom_feats
        for conv in self.convs:
            neighborhood = torch.relu(adjacency @ conv(h))
            # Skip connection only once dimensions match (i.e. after layer 1).
            h = neighborhood + h if (self.residual and h.shape == neighborhood.shape) else neighborhood
        denom = atom_mask.sum(1, keepdim=True).clamp(min=1.0)
        return (h * atom_mask.unsqueeze(-1)).sum(1) / denom  # masked mean pool


class SequenceEncoder(nn.Module):
    """Learned features: a small Transformer encoder over SMILES characters."""

    def __init__(
        self,
        hidden: int = 64,
        d_model: int = 48,
        layers: int = 1,
        nhead: int = 4,
        dim_feedforward: int = 96,
        dropout: float = 0.1,
        max_len: int = MAX_LEN,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False avoids a prototype-API warning on padded input.
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.project = nn.Linear(d_model, hidden)

    def forward(self, tokens: torch.Tensor, token_pad: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        h = self.encoder(h, src_key_padding_mask=token_pad)
        keep = (~token_pad).float().unsqueeze(-1)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # masked mean pool
        return self.project(pooled)


class GatedFusion(nn.Module):
    """Learned attention over branch embeddings.

    Carries obelus contracts, so Gate 1 exercises real shape and invariant
    checking on this architecture. The parameter is named ``x`` because
    ``icontract`` binds condition lambdas by argument name.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gate = nn.Linear(hidden, 1)

    @check_shapes("batch branches hidden -> batch hidden")
    @check_invariants(no_nans=True, no_infs=True, max_norm_ratio=10.0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=1)  # (B, K, 1)
        return (weights * x).sum(1)


class HybridMolecularClassifier(nn.Module):
    """Multi-branch molecular classifier; every branch is optional.

    Hydra instantiates the branches recursively and passes them in. A knocked-out
    branch arrives as ``nn.Identity`` and is skipped; a knocked-out ``fusion``
    falls back to an unlearned mean over the surviving branch embeddings.
    """

    def __init__(
        self,
        descriptor_branch: Optional[nn.Module] = None,
        graph_encoder: Optional[nn.Module] = None,
        sequence_encoder: Optional[nn.Module] = None,
        fusion: Optional[nn.Module] = None,
        hidden: int = 64,
    ):
        super().__init__()
        self.descriptor_branch = descriptor_branch
        self.graph_encoder = graph_encoder
        self.sequence_encoder = sequence_encoder
        self.fusion = fusion
        self.head = nn.Linear(hidden, 1)
        if not any(
            _present(m) for m in (descriptor_branch, graph_encoder, sequence_encoder)
        ):
            raise ValueError("at least one branch must remain active")

    def forward(
        self,
        descriptors: torch.Tensor,
        atom_feats: torch.Tensor,
        adjacency: torch.Tensor,
        atom_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_pad: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = []
        if _present(self.descriptor_branch):
            embeddings.append(self.descriptor_branch(descriptors))
        if _present(self.graph_encoder):
            embeddings.append(self.graph_encoder(atom_feats, adjacency, atom_mask))
        if _present(self.sequence_encoder):
            embeddings.append(self.sequence_encoder(tokens, token_pad))

        stacked = torch.stack(embeddings, dim=1)  # (B, K, hidden)
        pooled = self.fusion(stacked) if _present(self.fusion) else stacked.mean(1)
        return self.head(pooled).squeeze(-1)
