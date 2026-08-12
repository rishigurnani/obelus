"""Three molecular classifiers with a common fit/predict interface.

- ``DescriptorMLP``    — relies on **2D computed features** (RDKit descriptors).
- ``MolGNN``           — **learns features** from the molecular graph.
- ``SmilesTransformer``— **learns features** from tokenized SMILES.

All three are ``nn.Module`` subclasses (so obelus's mutator can corrupt the
baseline in place) and share one training loop. Each owns its own featurizer, so
the scorer stays architecture-agnostic: ``model.predict_proba(smiles)``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

# 2D descriptors for the feature-based baseline (name -> RDKit function).
_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("RingCount", Descriptors.RingCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("LabuteASA", Descriptors.LabuteASA),
    ("MolMR", Descriptors.MolMR),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("qed", Descriptors.qed),
]
N_DESCRIPTORS = len(_DESCRIPTORS)

# Atom-type vocabulary for the GNN (common drug elements + "other").
_ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
MAX_ATOMS = 72


def _mols(smiles: list[str]) -> list[Chem.Mol]:
    return [Chem.MolFromSmiles(s) for s in smiles]


class MoleculeClassifier(nn.Module):
    """Shared training/inference skeleton; subclasses provide featurizer + forward."""

    def __init__(self, *, epochs: int, lr: float, batch_size: int, seed: int = 0):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    # --- subclass hooks ---------------------------------------------------
    def prepare(self, smiles: list[str]) -> None:
        """Fit any featurizer state (scaler, vocab, sizes) on the training set."""

    def build(self) -> None:
        """Create the learnable submodules once feature dimensions are known."""
        raise NotImplementedError

    def featurize(self, smiles: list[str]) -> tuple[torch.Tensor, ...]:
        """Return a tuple of batched tensors consumed positionally by ``forward``."""
        raise NotImplementedError

    # --- shared training / inference -------------------------------------
    def fit(self, smiles: list[str], y: np.ndarray) -> "MoleculeClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.prepare(smiles)
        self.build()
        tensors = self.featurize(smiles)
        target = torch.tensor(np.asarray(y), dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        n = len(smiles)
        order = np.arange(n)
        self.train()
        for _ in range(self.epochs):
            np.random.shuffle(order)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                sub = tuple(t[batch] for t in tensors)
                optimizer.zero_grad()
                loss = loss_fn(self.forward(*sub), target[batch])
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        self.eval()
        logits = self.forward(*self.featurize(smiles))
        return torch.sigmoid(logits).cpu().numpy()


class DescriptorMLP(MoleculeClassifier):
    """Feature-based baseline: standardized RDKit 2D descriptors -> MLP."""

    def __init__(self, *, seed: int = 0):
        super().__init__(epochs=120, lr=1e-3, batch_size=128, seed=seed)
        self._scaler: StandardScaler | None = None

    @staticmethod
    def _raw(smiles: list[str]) -> np.ndarray:
        rows = []
        for mol in _mols(smiles):
            if mol is None:
                rows.append([0.0] * N_DESCRIPTORS)
                continue
            rows.append([fn(mol) for _, fn in _DESCRIPTORS])
        return np.nan_to_num(np.asarray(rows, dtype=np.float32))

    def prepare(self, smiles: list[str]) -> None:
        self._scaler = StandardScaler().fit(self._raw(smiles))

    def build(self) -> None:
        self.net = nn.Sequential(
            nn.Linear(N_DESCRIPTORS, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def featurize(self, smiles: list[str]) -> tuple[torch.Tensor, ...]:
        x = self._scaler.transform(self._raw(smiles))
        return (torch.tensor(x, dtype=torch.float32),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _atom_features(atom: Chem.Atom) -> list[float]:
    onehot = [1.0 if atom.GetAtomicNum() == e else 0.0 for e in _ELEMENTS]
    onehot.append(0.0 if atom.GetAtomicNum() in _ELEMENTS else 1.0)  # "other"
    return onehot + [
        atom.GetDegree() / 4.0,
        atom.GetFormalCharge(),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetTotalNumHs() / 4.0,
    ]


_ATOM_FEAT_DIM = len(_ELEMENTS) + 1 + 5


class MolGNN(MoleculeClassifier):
    """Learns features via a 2-layer normalized graph convolution over atoms."""

    def __init__(self, *, seed: int = 0, hidden: int = 64):
        super().__init__(epochs=60, lr=5e-3, batch_size=128, seed=seed)
        self.hidden = hidden

    def build(self) -> None:
        self.gc1 = nn.Linear(_ATOM_FEAT_DIM, self.hidden)
        self.gc2 = nn.Linear(self.hidden, self.hidden)
        self.readout = nn.Sequential(nn.Linear(self.hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def featurize(self, smiles: list[str]) -> tuple[torch.Tensor, ...]:
        n = len(smiles)
        X = np.zeros((n, MAX_ATOMS, _ATOM_FEAT_DIM), dtype=np.float32)
        A = np.zeros((n, MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
        mask = np.zeros((n, MAX_ATOMS), dtype=np.float32)
        for i, mol in enumerate(_mols(smiles)):
            if mol is None:
                continue
            k = min(mol.GetNumAtoms(), MAX_ATOMS)
            adj = np.eye(k, dtype=np.float32)  # self-loops
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a < k and b < k:
                    adj[a, b] = adj[b, a] = 1.0
            deg = adj.sum(1)
            dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
            A[i, :k, :k] = dinv[:, None] * adj * dinv[None, :]  # symmetric-normalized
            for j, atom in enumerate(mol.GetAtoms()):
                if j >= k:
                    break
                X[i, j] = _atom_features(atom)
            mask[i, :k] = 1.0
        return (
            torch.from_numpy(X),
            torch.from_numpy(A),
            torch.from_numpy(mask),
        )

    def forward(self, X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.relu(A @ self.gc1(X))
        h = torch.relu(A @ self.gc2(h))
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / denom  # masked mean pool
        return self.readout(pooled).squeeze(-1)


class SmilesTransformer(MoleculeClassifier):
    """Learns features via a small Transformer encoder over SMILES characters."""

    def __init__(self, *, seed: int = 0, d_model: int = 64, max_len: int = 100):
        super().__init__(epochs=25, lr=5e-4, batch_size=128, seed=seed)
        self.d_model = d_model
        self.max_len = max_len
        self._vocab: dict[str, int] = {}

    def prepare(self, smiles: list[str]) -> None:
        chars = sorted({c for s in smiles for c in s})
        # 0 = PAD, 1 = UNK, then the observed characters.
        self._vocab = {c: i + 2 for i, c in enumerate(chars)}
        self.max_len = min(self.max_len, max(len(s) for s in smiles))

    def build(self) -> None:
        vocab_size = len(self._vocab) + 2
        self.embed = nn.Embedding(vocab_size, self.d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        # enable_nested_tensor=False avoids a prototype-API warning on padded input.
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(self.d_model, 1)

    def featurize(self, smiles: list[str]) -> tuple[torch.Tensor, ...]:
        ids = np.zeros((len(smiles), self.max_len), dtype=np.int64)
        pad = np.ones((len(smiles), self.max_len), dtype=bool)  # True == padding
        for i, s in enumerate(smiles):
            for j, c in enumerate(s[: self.max_len]):
                ids[i, j] = self._vocab.get(c, 1)
                pad[i, j] = False
        return (torch.from_numpy(ids), torch.from_numpy(pad))

    def forward(self, ids: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids) + self.pos[:, : ids.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # masked mean pool
        return self.head(pooled).squeeze(-1)


_ARCHITECTURES = {
    "descriptor_mlp": DescriptorMLP,
    "gnn": MolGNN,
    "transformer": SmilesTransformer,
}


def build_model(arch: str, *, seed: int = 0) -> MoleculeClassifier:
    """Construct an untrained classifier by architecture name."""
    if arch not in _ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; choose from {list(_ARCHITECTURES)}")
    return _ARCHITECTURES[arch](seed=seed)
