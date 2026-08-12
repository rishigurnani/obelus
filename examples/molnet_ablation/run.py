"""Gate three molecular architectures on BACE-1 with obelus.

Question obelus answers here: *are the feature-learning models (a graph net and
a SMILES transformer) non-inferior to a strong 2D-descriptor baseline across
every failure-mode slice?* A single significant F1 degradation on any slice
rejects an architecture.

Run:  python -m examples.molnet_ablation.run   (needs the ``examples`` extra)
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import f1_score

from examples.molnet_ablation.data import SLICE_DOC, build_dataset
from examples.molnet_ablation.models import N_DESCRIPTORS, build_model
from obelus import run_ablation


def _make_factory(train_smiles, train_y, timings):
    """Return a ModelFactory that builds and trains a model once per variant."""

    def factory(overrides):
        arch = overrides.get("arch", "descriptor_mlp")
        t0 = time.perf_counter()
        model = build_model(arch, seed=0).fit(train_smiles, train_y)
        timings[arch] = time.perf_counter() - t0
        return model

    return factory


def _make_scorer(dataset):
    """Return a Scorer: F1 on a bootstrap resample of a slice, per fold.

    Predictions for a given (model, slice) are computed once and cached; each
    fold is a seeded bootstrap of the slice, giving the paired variance the
    permutation test needs. Baseline and variant share the same resample.
    """
    cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def scorer(model, slice_name, fold):
        key = (id(model), slice_name)
        if key not in cache:
            idx = dataset.slices[slice_name]
            smiles, labels = dataset.subset(idx)
            cache[key] = (model.predict_proba(smiles), labels)
        probs, labels = cache[key]
        rng = np.random.default_rng(1000 + fold)
        boot = rng.integers(0, len(labels), len(labels))
        preds = (probs[boot] >= 0.5).astype(int)
        return f1_score(labels[boot], preds, zero_division=0)

    return scorer


def _verdict(is_non_inferior: bool, baseline_mean: float, variant_mean: float) -> str:
    """Three-way display label from a (one-tailed) degradation test's outputs.

    ``evaluate_non_inferiority`` only tests for degradation, by design (obelus's
    gate is zero-tolerance on degradation, not a symmetric A/B test). So
    "non-inferior" covers both "actually better" and "roughly the same" — this
    splits those on the sign of the mean gap for a more legible printout. It is
    *not* a claim that the improvement itself is statistically significant.
    """
    if not is_non_inferior:
        return "DEGRADED"
    return "IMPROVED" if variant_mean > baseline_mean else "ok"


def main() -> None:
    start = time.perf_counter()
    dataset = build_dataset(seed=0)

    print("BACE-1 active/inactive gating  —  baseline: descriptor_mlp\n")
    print(f"training molecules: {len(dataset.train_idx)}")
    print("failure-mode slices:")
    for name, idx in dataset.slices.items():
        pos = int(dataset.labels[idx].sum())
        print(f"  {name:16s} n={len(idx):4d}  active={pos:3d}  — {SLICE_DOC[name]}")
    print()

    train_smiles, train_y = dataset.subset(dataset.train_idx)
    timings: dict[str, float] = {}

    result = run_ablation(
        scorer=_make_scorer(dataset),
        model_factory=_make_factory(train_smiles, train_y, timings),
        variants={"gnn": {"arch": "gnn"}, "smiles_transformer": {"arch": "transformer"}},
        slices=list(dataset.slices),
        input_shape=(8, N_DESCRIPTORS),  # baseline descriptor MLP, for gates 1-2
        # Models train once, so "folds" are seeded evaluation bootstraps that give
        # the paired permutation test its variance. 10 -> 2^10 arrangements, far
        # finer p-value resolution than 5 folds (2^5=32, min p=0.031).
        cv_folds=10,
        sanity_mutation=True,
        mutator_fraction=0.5,
        p_alpha=0.05,
        greater_is_better=True,  # F1: higher is better
        seed=0,
    )

    print("verification ladder:")
    for gate in result.report.results:
        status = "PASS" if gate.passed else "FAIL"
        print(f"  [{status}] {gate.name:16s} {gate.summary}")
    fire = result.report.get("fire_drill")
    if fire is not None:
        pvals = ", ".join(f"{s}={p:.3f}" for s, p in fire.data["p_degrade"].items())
        print(f"           slice sensitivity p_degrade: {pvals}")
    print()

    print("non-inferiority decisions (mean F1 across folds):")
    df = result.summary
    if df.empty:
        print(f"  ladder halted at '{result.report.halted_at}'; no decisions produced")
    else:
        header = f"  {'variant':18s} {'slice':16s} {'base_F1':>8s} {'var_F1':>8s} {'p_deg':>7s}  verdict"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in df.iterrows():
            verdict = _verdict(row["is_non_inferior"], row["baseline_mean"], row["variant_mean"])
            print(
                f"  {row['variant']:18s} {row['slice']:16s} "
                f"{row['baseline_mean']:8.3f} {row['variant_mean']:8.3f} "
                f"{row['p_degrade']:7.3f}  {verdict}"
            )
        print(
            "\n  verdict key: DEGRADED = p_degrade < alpha (significant);"
            " IMPROVED/ok = non-inferior, split by mean F1 (directional only —"
            " obelus's gate is one-tailed and never tests improvement for"
            " significance)."
        )
        print()
        for variant, decision in result.report.get("non_inferiority").data["decisions"].items():
            print(f"  => {variant}: {decision}")

    print("\ntraining time per architecture (s): " +
          ", ".join(f"{a}={t:.1f}" for a, t in timings.items()))
    print(f"total wall time: {time.perf_counter() - start:.1f}s")
    df.to_csv("example.csv")


if __name__ == "__main__":
    main()
