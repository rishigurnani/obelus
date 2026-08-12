"""Fast smoke tests for the BACE example (no training, so the suite stays quick)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from omegaconf import OmegaConf

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

from examples.molnet_ablation.data import SLICE_DOC, build_dataset
from examples.molnet_ablation.features import MolFeaturizer, example_batch
from examples.molnet_ablation.layers import HybridMolecularClassifier
from examples.molnet_ablation.run import CONFIG_PATH, PROJECT_PREFIX
from examples.molnet_ablation.training import fit, predict_proba
from obelus.adapters import HydraModelFactory
from obelus.core.discovery import generate_knockouts

EXPECTED_KNOCKOUTS = {
    "no_descriptor_branch",
    "no_graph_encoder",
    "no_sequence_encoder",
    "no_fusion",
    "no_residual",
}


def _cfg():
    return OmegaConf.load(CONFIG_PATH)


def test_dataset_slices_are_disjoint_and_documented():
    ds = build_dataset(seed=0)
    assert len(ds.train_idx) > 200
    seen: set[int] = set()
    for name, idx in ds.slices.items():
        assert name in SLICE_DOC
        assert len(idx) > 0
        assert not (set(idx.tolist()) & seen), f"slice {name} overlaps another"
        seen.update(idx.tolist())
    assert not (set(ds.train_idx.tolist()) & seen)  # slices held out from training


def test_labels_are_binary():
    ds = build_dataset(seed=0)
    assert set(np.unique(ds.labels).tolist()) <= {0, 1}


def test_config_yields_the_expected_ablation_matrix():
    """The variant list is derived from the YAML, never hand-written."""
    knockouts = generate_knockouts(_cfg(), custom_prefixes=(PROJECT_PREFIX,))
    assert set(knockouts) == EXPECTED_KNOCKOUTS | {"baseline"}
    assert knockouts["no_fusion"] == {"model.fusion._target_": "torch.nn.Identity"}
    assert knockouts["no_residual"] == {"model.graph_encoder.residual.active": False}


def test_library_layers_are_not_ablated():
    # torch.nn.* targets must never be knocked out — only project modules.
    knockouts = generate_knockouts(_cfg(), custom_prefixes=(PROJECT_PREFIX,))
    for overrides in knockouts.values():
        assert all("torch.nn" not in path for path in overrides)


@pytest.mark.parametrize("variant", sorted(EXPECTED_KNOCKOUTS))
def test_every_knockout_instantiates_and_runs_forward(variant):
    """Each config-level ablation must produce a working architecture."""
    cfg = _cfg()
    knockouts = generate_knockouts(cfg, custom_prefixes=(PROJECT_PREFIX,))
    model = HydraModelFactory(cfg)(knockouts[variant])
    assert isinstance(model, HybridMolecularClassifier)
    out = model(*example_batch(3).as_args())
    assert out.shape == (3,)


def test_knockout_actually_removes_the_module():
    cfg = _cfg()
    knockouts = generate_knockouts(cfg, custom_prefixes=(PROJECT_PREFIX,))
    ablated = HydraModelFactory(cfg)(knockouts["no_graph_encoder"])
    baseline = HydraModelFactory(cfg)({})
    assert isinstance(ablated.graph_encoder, nn.Identity)
    assert not isinstance(baseline.graph_encoder, nn.Identity)


def test_residual_toggle_flows_into_the_encoder():
    cfg = _cfg()
    knockouts = generate_knockouts(cfg, custom_prefixes=(PROJECT_PREFIX,))
    off = HydraModelFactory(cfg)(knockouts["no_residual"])
    on = HydraModelFactory(cfg)({})
    assert off.graph_encoder.residual is False
    assert on.graph_encoder.residual is True


def test_tiny_fit_then_predict_produces_probabilities():
    ds = build_dataset(seed=0)
    smiles, labels = ds.subset(ds.train_idx[:48])
    featurizer = MolFeaturizer().fit(smiles)
    batch = featurizer.transform(smiles)
    model = HydraModelFactory(_cfg())({})
    fit(model, batch, labels, epochs=1)
    probs = predict_proba(model, batch)
    assert probs.shape == (48,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_featurizer_caches_repeated_batches():
    ds = build_dataset(seed=0)
    smiles, _ = ds.subset(ds.train_idx[:32])
    featurizer = MolFeaturizer().fit(smiles)
    assert featurizer.transform(smiles) is featurizer.transform(smiles)
