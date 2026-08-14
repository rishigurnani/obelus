"""BACE-1 dataset loading and failure-mode slice construction.

The dataset is ~1500 small molecules labeled active/inactive against human
β-secretase 1 (BACE-1). We train each architecture **once** on a shared training
pool, then probe it on *failure-mode cohorts* — each a distinct way a molecular
classifier is known to break — held out from that pool. Cohorts overlap, so each
one measures its own failure mode rather than "that mode, minus whatever an
earlier cohort already claimed"; only ``in_distribution`` is exclusive, being the
control for "no failure mode at all".
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bace.csv"

# How each failure-mode cohort is defined and why it stresses a real weakness.
SLICE_DOC = {
    "in_distribution": "random held-out molecules — the in-distribution control",
    "novel_scaffold": "unique Murcko scaffolds unseen in training (structural novelty)",
    "large": "top-decile heavy-atom count (molecular-size extrapolation)",
    "lipophilic": "top-decile logP (physicochemical covariate shift)",
    "flexible": "top-decile rotatable bonds (conformational flexibility)",
}


@dataclass
class Dataset:
    smiles: list[str]
    labels: np.ndarray  # (N,) int 0/1
    train_idx: np.ndarray
    slices: dict[str, np.ndarray]  # name -> indices into smiles/labels

    def subset(self, idx: np.ndarray) -> tuple[list[str], np.ndarray]:
        return [self.smiles[i] for i in idx], self.labels[idx]


def _load_rows(path: Path) -> tuple[list[str], np.ndarray, list[Chem.Mol]]:
    smiles: list[str] = []
    labels: list[int] = []
    mols: list[Chem.Mol] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            smiles.append(row["smiles"])
            labels.append(int(row["label"]))
            mols.append(mol)
    return smiles, np.asarray(labels), mols


def build_dataset(
    seed: int = 0,
    *,
    holdout_frac: float = 0.15,
    cohort_cap: int = 150,
    tail_support: float = 0.5,
    path: Path = DATA_PATH,
) -> Dataset:
    """Load BACE and carve overlapping failure-mode slices out of a shared pool."""
    smiles, labels, mols = _load_rows(path)
    n = len(smiles)
    rng = np.random.default_rng(seed)

    heavy = np.asarray([m.GetNumHeavyAtoms() for m in mols])
    logp = np.asarray([Descriptors.MolLogP(m) for m in mols])
    rotb = np.asarray([Descriptors.NumRotatableBonds(m) for m in mols])
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    scaffold_counts = Counter(scaffolds)

    assigned = np.zeros(n, dtype=bool)
    slices: dict[str, np.ndarray] = {}

    def take(name: str, mask: np.ndarray, cap: int, *, support: float = 0.0) -> None:
        candidates = np.where(mask)[0]
        rng.shuffle(candidates)
        # Leave `support` of the tail in training. Holding a cohort out *entirely*
        # (0 of the 162 largest molecules were ever seen) forces pure
        # extrapolation, and where label prevalence also flips across the cut —
        # 0.64 active above the size threshold vs 0.43 below — the model learns
        # the relationship backwards and scores negative MCC on its own cohort.
        cap = min(cap, int(round(len(candidates) * (1.0 - support))))
        chosen = candidates[:cap]
        slices[name] = np.sort(chosen)
        assigned[chosen] = True  # union of every cohort is held out of training

    # Cohorts OVERLAP: a large, flexible molecule belongs in both, and forcing a
    # winner would silently redefine `flexible` as "flexible but small" — the
    # slice would no longer measure the failure mode it is named after. Sharing
    # molecules makes the per-slice tests dependent, which is exactly what
    # obelus.core.stats.select_correction detects.
    singleton = np.asarray([scaffold_counts[s] == 1 for s in scaffolds])
    take("novel_scaffold", singleton, cohort_cap, support=tail_support)
    take("large", heavy >= np.quantile(heavy, 0.9), cohort_cap, support=tail_support)
    take("lipophilic", logp >= np.quantile(logp, 0.9), cohort_cap, support=tail_support)
    take("flexible", rotb >= np.quantile(rotb, 0.9), cohort_cap, support=tail_support)

    remaining = np.where(~assigned)[0]
    rng.shuffle(remaining)
    n_holdout = int(holdout_frac * n)
    take("in_distribution", np.isin(np.arange(n), remaining[:n_holdout]), n_holdout)

    train_idx = np.sort(np.where(~assigned)[0])
    return Dataset(smiles=smiles, labels=labels, train_idx=train_idx, slices=slices)
