"""Molecular featurization: RDKit descriptors, atom graphs, and SMILES tokens.

Kept separate from the model so the network stays a pure tensor -> tensor
``nn.Module`` — which is what lets Hydra instantiate it from config alone and
lets obelus's gates 1-2 drive it with synthetic inputs.

Featurization is expensive and identical across every knockout variant, so a
fitted :class:`MolFeaturizer` caches each batch it builds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

# --- 2D computed descriptors (the hand-crafted feature branch) --------------
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

# --- graph branch -----------------------------------------------------------
_ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I
ATOM_FEAT_DIM = len(_ELEMENTS) + 1 + 5  # one-hot + "other" + 5 scalar props
MAX_ATOMS = 64  # covers >99% of BACE molecules

# --- sequence branch --------------------------------------------------------
# A fixed vocabulary keeps the architecture config self-contained: vocab size is
# a constant, not something learned from whichever split you happen to load.
SMILES_CHARS = "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVWXYZ[\\]abcdefgilmnoprstuy"
SMILES_VOCAB = {c: i + 2 for i, c in enumerate(SMILES_CHARS)}  # 0=PAD, 1=UNK
VOCAB_SIZE = len(SMILES_VOCAB) + 2
MAX_LEN = 96


@dataclass(frozen=True)
class MolBatch:
    """Every representation of a molecule batch, consumed branch-by-branch."""

    descriptors: torch.Tensor  # (N, N_DESCRIPTORS) float
    atom_feats: torch.Tensor  # (N, MAX_ATOMS, ATOM_FEAT_DIM) float
    adjacency: torch.Tensor  # (N, MAX_ATOMS, MAX_ATOMS) float, normalized
    atom_mask: torch.Tensor  # (N, MAX_ATOMS) float
    tokens: torch.Tensor  # (N, MAX_LEN) long
    token_pad: torch.Tensor  # (N, MAX_LEN) bool, True == padding

    def __len__(self) -> int:
        return self.descriptors.shape[0]

    def select(self, idx) -> "MolBatch":
        """Row-subset every tensor (used for mini-batching)."""
        return MolBatch(
            self.descriptors[idx],
            self.atom_feats[idx],
            self.adjacency[idx],
            self.atom_mask[idx],
            self.tokens[idx],
            self.token_pad[idx],
        )

    def as_args(self) -> tuple[torch.Tensor, ...]:
        """Positional forward() arguments, in the model's declared order."""
        return (
            self.descriptors,
            self.atom_feats,
            self.adjacency,
            self.atom_mask,
            self.tokens,
            self.token_pad,
        )


def _atom_features(atom: Chem.Atom) -> list[float]:
    onehot = [1.0 if atom.GetAtomicNum() == e else 0.0 for e in _ELEMENTS]
    onehot.append(0.0 if atom.GetAtomicNum() in _ELEMENTS else 1.0)
    return onehot + [
        atom.GetDegree() / 4.0,
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetTotalNumHs() / 4.0,
    ]


class MolFeaturizer:
    """Builds :class:`MolBatch` objects; fit the descriptor scaler on train only."""

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._cache: dict[tuple[str, ...], MolBatch] = {}

    @staticmethod
    def _raw_descriptors(mols: list[Chem.Mol]) -> np.ndarray:
        rows = [
            [fn(mol) for _, fn in _DESCRIPTORS] if mol is not None else [0.0] * N_DESCRIPTORS
            for mol in mols
        ]
        return np.nan_to_num(np.asarray(rows, dtype=np.float32))

    def fit(self, smiles: list[str]) -> "MolFeaturizer":
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        self._scaler = StandardScaler().fit(self._raw_descriptors(mols))
        return self

    def transform(self, smiles: list[str]) -> MolBatch:
        key = tuple(smiles)
        if key in self._cache:
            return self._cache[key]
        if self._scaler is None:
            raise RuntimeError("call fit() on the training molecules first")

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        n = len(mols)
        desc = self._scaler.transform(self._raw_descriptors(mols)).astype(np.float32)

        X = np.zeros((n, MAX_ATOMS, ATOM_FEAT_DIM), dtype=np.float32)
        A = np.zeros((n, MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
        mask = np.zeros((n, MAX_ATOMS), dtype=np.float32)
        ids = np.zeros((n, MAX_LEN), dtype=np.int64)
        pad = np.ones((n, MAX_LEN), dtype=bool)

        for i, (mol, smi) in enumerate(zip(mols, smiles)):
            for j, c in enumerate(smi[:MAX_LEN]):
                ids[i, j] = SMILES_VOCAB.get(c, 1)
                pad[i, j] = False
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
            A[i, :k, :k] = dinv[:, None] * adj * dinv[None, :]  # symmetric norm
            for j, atom in enumerate(mol.GetAtoms()):
                if j >= k:
                    break
                X[i, j] = _atom_features(atom)
            mask[i, :k] = 1.0

        batch = MolBatch(
            descriptors=torch.from_numpy(desc),
            atom_feats=torch.from_numpy(X),
            adjacency=torch.from_numpy(A),
            atom_mask=torch.from_numpy(mask),
            tokens=torch.from_numpy(ids),
            token_pad=torch.from_numpy(pad),
        )
        self._cache[key] = batch
        return batch


def example_batch(n: int = 4) -> MolBatch:
    """A synthetic batch with correct shapes/dtypes, for obelus gates 1-2."""
    return MolBatch(
        descriptors=torch.randn(n, N_DESCRIPTORS),
        atom_feats=torch.randn(n, MAX_ATOMS, ATOM_FEAT_DIM),
        adjacency=torch.eye(MAX_ATOMS).expand(n, MAX_ATOMS, MAX_ATOMS).contiguous(),
        atom_mask=torch.ones(n, MAX_ATOMS),
        tokens=torch.randint(0, VOCAB_SIZE, (n, MAX_LEN)),
        token_pad=torch.zeros(n, MAX_LEN, dtype=torch.bool),
    )
