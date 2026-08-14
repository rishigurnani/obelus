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
from examples.molnet_ablation.run import CONFIG_PATH
from examples.molnet_ablation.training import fit, predict_proba
from obelus.adapters import HydraModelFactory
from obelus.core.options import COMPLEXIFY, SIMPLIFY, generate_moves

SIMPLIFY_MOVES = {"no_descriptor_branch", "no_graph_encoder"}
COMPLEXIFY_MOVES = {"add_sequence_encoder", "add_fusion"}


def _cfg():
    """The ordered-options config — the only form obelus supports."""
    return OmegaConf.load(CONFIG_PATH)


def test_dataset_slices_overlap_and_are_documented():
    """Failure-mode cohorts overlap on purpose; only training stays disjoint."""
    ds = build_dataset(seed=0)
    assert len(ds.train_idx) > 200
    seen: set[int] = set()
    for name, idx in ds.slices.items():
        assert name in SLICE_DOC
        assert len(idx) > 0
        seen.update(idx.tolist())
    # Big molecules are also flexible: forcing one cohort to win would redefine
    # `flexible` as "flexible but small". The overlap is what makes the
    # per-slice tests dependent, and so what selects the Holm correction.
    assert set(ds.slices["large"].tolist()) & set(ds.slices["flexible"].tolist())
    assert not (set(ds.train_idx.tolist()) & seen)  # slices held out from training


def test_labels_are_binary():
    ds = build_dataset(seed=0)
    assert set(np.unique(ds.labels).tolist()) <= {0, 1}


@pytest.mark.parametrize("move", sorted(SIMPLIFY_MOVES))
def test_every_ablation_instantiates_and_runs_forward(move):
    """Each config-level ablation must produce a working architecture."""
    cfg = _cfg()
    model = HydraModelFactory(cfg)(generate_moves(cfg, SIMPLIFY)[move])
    assert isinstance(model, HybridMolecularClassifier)
    assert model(*example_batch(3).as_args()).shape == (3,)


def test_ablation_actually_removes_the_module():
    cfg = _cfg()
    moves = generate_moves(cfg, SIMPLIFY)
    ablated = HydraModelFactory(cfg)(moves["no_graph_encoder"])
    baseline = HydraModelFactory(cfg)(moves["baseline"])
    assert isinstance(ablated.graph_encoder, nn.Identity)
    assert not isinstance(baseline.graph_encoder, nn.Identity)


def test_residual_flag_flows_into_the_encoder():
    """A plain constructor flag now, not an `active:` toggle discovery flips."""
    cfg = _cfg()
    on = HydraModelFactory(cfg)(generate_moves(cfg, SIMPLIFY)["baseline"])
    assert on.graph_encoder.residual is True


def test_a_config_without_options_is_rejected():
    """One standard: a bare _target_ config is an error, not a silent fallback."""
    cfg = OmegaConf.create({"model": {"_target_": "torch.nn.Identity"}})
    with pytest.raises(ValueError, match="no _options_ nodes found"):
        generate_moves(cfg, SIMPLIFY)


def test_tiny_fit_then_predict_produces_probabilities():
    ds = build_dataset(seed=0)
    smiles, labels = ds.subset(ds.train_idx[:48])
    featurizer = MolFeaturizer().fit(smiles)
    batch = featurizer.transform(smiles)
    cfg = _cfg()
    model = HydraModelFactory(cfg)(generate_moves(cfg, SIMPLIFY)["baseline"])
    fit(model, batch, labels, epochs=1)
    probs = predict_proba(model, batch)
    assert probs.shape == (48,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_featurizer_caches_repeated_batches():
    ds = build_dataset(seed=0)
    smiles, _ = ds.subset(ds.train_idx[:32])
    featurizer = MolFeaturizer().fit(smiles)
    assert featurizer.transform(smiles) is featurizer.transform(smiles)


def test_one_config_drives_both_directions():
    """One config, two experiments — the point of the ordered-options form."""
    cfg = _cfg()
    assert set(generate_moves(cfg, SIMPLIFY)) == SIMPLIFY_MOVES | {"baseline"}
    assert set(generate_moves(cfg, COMPLEXIFY)) == COMPLEXIFY_MOVES | {"baseline"}


@pytest.mark.parametrize("move", sorted(COMPLEXIFY_MOVES | {"baseline"}))
def test_every_insertion_variant_instantiates_and_runs(move):
    cfg = _cfg()
    model = HydraModelFactory(cfg)(generate_moves(cfg, COMPLEXIFY)[move])
    assert model(*example_batch(3).as_args()).shape == (3,)


def test_option_scaffolding_never_reaches_instantiate():
    """_options_/_current_ must be replaced, not merged around."""
    from examples.molnet_ablation.layers import GatedFusion

    cfg = _cfg()
    model = HydraModelFactory(cfg)(generate_moves(cfg, COMPLEXIFY)["add_fusion"])
    assert isinstance(model.fusion, GatedFusion)
    assert isinstance(model.sequence_encoder, nn.Identity)  # still at rank 0
