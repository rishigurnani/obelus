"""Fast smoke tests for the BACE example (no training, so the suite stays quick)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

from examples.molnet_ablation.data import SLICE_DOC, build_dataset
from examples.molnet_ablation.models import N_DESCRIPTORS, build_model


def test_dataset_slices_are_disjoint_and_documented():
    ds = build_dataset(seed=0)
    assert len(ds.train_idx) > 200
    seen: set[int] = set()
    for name, idx in ds.slices.items():
        assert name in SLICE_DOC
        assert len(idx) > 0
        assert not (set(idx.tolist()) & seen), f"slice {name} overlaps another"
        seen.update(idx.tolist())
    # slices are held out from the training pool
    assert not (set(ds.train_idx.tolist()) & seen)


def test_labels_are_binary():
    ds = build_dataset(seed=0)
    assert set(np.unique(ds.labels).tolist()) <= {0, 1}


@pytest.mark.parametrize("arch", ["descriptor_mlp", "gnn", "transformer"])
def test_each_architecture_featurizes_and_predicts_after_a_tiny_fit(arch):
    ds = build_dataset(seed=0)
    smiles, labels = ds.subset(ds.train_idx[:64])
    model = build_model(arch, seed=0)
    model.epochs = 1  # keep the smoke test fast
    model.fit(smiles, labels)
    probs = model.predict_proba(smiles[:8])
    assert probs.shape == (8,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_descriptor_count_matches_input_shape_contract():
    # The baseline's descriptor width is what run.py feeds gates 1-2 as input_shape.
    assert N_DESCRIPTORS == 16
