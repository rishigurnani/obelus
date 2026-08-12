"""Config-driven architecture ablation on BACE-1, gated by obelus.

Everything ablated here comes from ``examples/configs/architecture.yaml``:
obelus reads the Hydra config, discovers each architectural decision, builds one
knockout variant per decision, trains it, and asks whether dropping that
component causes a *statistically significant* F1 loss on any failure-mode
slice. No variant list is written by hand — add a module to the YAML and it is
ablated on the next run.

Run:  python -m examples.molnet_ablation.run   (needs the ``examples`` extra)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from examples.molnet_ablation.data import SLICE_DOC, build_dataset
from examples.molnet_ablation.features import MolFeaturizer, example_batch
from examples.molnet_ablation.training import fit, predict_proba
from obelus import run_ablation
from obelus.adapters import HydraModelFactory
from obelus.core.discovery import generate_knockouts
from obelus.core.power import min_achievable_p

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "architecture.yaml"
# Discovery ablates modules whose _target_ starts with the project's own package
# (library layers like torch.nn.* are left alone).
PROJECT_PREFIX = "examples.molnet_ablation."

P_ALPHA = 0.05
CV_FOLDS = 10
# The smallest F1 drop worth caring about. Declaring this up front is what makes
# a non-significant result interpretable: without it, "no difference" cannot be
# distinguished from "no ability to see a difference".
PRACTICAL_MARGIN = 0.03
TARGET_POWER = 0.8


def _make_factory(cfg, knockouts, train_batch, train_y, timings, seed=0):
    """Hydra builds the architecture; this wrapper trains what Hydra returns."""
    hydra_factory = HydraModelFactory(cfg)
    # Reverse-map an override set back to its variant name, for readable timings.
    names = {frozenset(ov.items()): name for name, ov in knockouts.items()}

    def factory(overrides):
        started = time.perf_counter()
        # Seed before instantiation, not just before training: weight init draws
        # from the global RNG, and the earlier gates consume a variable amount of
        # it (hypothesis fuzzing). Without this, every variant would be compared
        # against a differently-initialized baseline and runs would not repeat.
        torch.manual_seed(seed)
        model = hydra_factory(overrides)  # real hydra.utils.instantiate
        fit(model, train_batch, train_y, seed=seed)
        label = names.get(frozenset(overrides.items()), "baseline")
        timings[label] = time.perf_counter() - started
        return model

    return factory


def _make_scorer(dataset, featurizer):
    """F1 on a seeded bootstrap of a slice, one resample per fold.

    Each model trains once, so predictions per (model, slice) are computed once
    and cached; folds resample them, giving the paired permutation test its
    variance. Baseline and variant always see the same resample for a fold.
    """
    cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def scorer(model, slice_name, fold):
        key = (id(model), slice_name)
        if key not in cache:
            smiles, labels = dataset.subset(dataset.slices[slice_name])
            cache[key] = (predict_proba(model, featurizer.transform(smiles)), labels)
        probs, labels = cache[key]
        boot = np.random.default_rng(1000 + fold).integers(0, len(labels), len(labels))
        return f1_score(labels[boot], (probs[boot] >= 0.5).astype(int), zero_division=0)

    return scorer


def _verdict(row) -> str:
    """Four-way label; every branch is backed by a test, including the null one.

    Both tails of the paired permutation test are used, so IMPROVED is a real
    claim (``p_improve < alpha``), not "the mean happened to be higher". And a
    non-significant result is only called ``same`` when the test actually had
    the power to detect the margin — otherwise it is INCONCLUSIVE, because
    absence of evidence is not evidence of absence.
    """
    if not row["is_non_inferior"]:
        return "DEGRADED"
    if row["is_improved"]:
        return "IMPROVED"
    return "same" if row["adequately_powered"] else "INCONCLUSIVE"


def main() -> None:
    start = time.perf_counter()

    cfg = OmegaConf.load(CONFIG_PATH)
    knockouts = generate_knockouts(cfg, custom_prefixes=(PROJECT_PREFIX,))

    dataset = build_dataset(seed=0)
    train_smiles, train_y = dataset.subset(dataset.train_idx)
    featurizer = MolFeaturizer().fit(train_smiles)
    train_batch = featurizer.transform(train_smiles)

    print("BACE-1 architecture ablation — config-driven by obelus\n")
    print(f"config: {CONFIG_PATH.relative_to(Path.cwd())}")
    print(f"training molecules: {len(dataset.train_idx)}\n")

    print("architectural decisions discovered from the Hydra config:")
    for name, overrides in knockouts.items():
        if name == "baseline":
            continue
        (path, value), = overrides.items()
        print(f"  {name:22s} {path} -> {value}")
    print()

    print("failure-mode slices:")
    for name, idx in dataset.slices.items():
        print(f"  {name:16s} n={len(idx):4d}  active={int(dataset.labels[idx].sum()):3d}"
              f"  — {SLICE_DOC[name]}")
    print()

    # Structural check, independent of any data: with n paired folds the test has
    # only 2**n sign-flip arrangements, so p can never fall below 2**-n. If that
    # floor exceeds alpha, every slice is guaranteed to pass — vacuously.
    floor = min_achievable_p(CV_FOLDS)
    print(f"test capability: {CV_FOLDS} folds -> smallest attainable p = {floor:.4f}"
          f" vs alpha = {P_ALPHA}"
          f"  [{'OK' if floor <= P_ALPHA else 'IMPOSSIBLE — cannot reject at this alpha'}]")
    print(f"  (5 folds would give {min_achievable_p(5):.4f}; 4 folds"
          f" {min_achievable_p(4):.4f} > {P_ALPHA}, i.e. structurally unable to reject)")
    print(f"declared meaningful degradation: {PRACTICAL_MARGIN:.3f} F1"
          f" at {TARGET_POWER:.0%} power\n")

    timings: dict[str, float] = {}
    result = run_ablation(
        cfg,
        scorer=_make_scorer(dataset, featurizer),
        model_factory=_make_factory(cfg, knockouts, train_batch, train_y, timings),
        variants=knockouts,
        slices=list(dataset.slices),
        # The model takes six tensors, so gates 1-2 need a real example batch
        # rather than a single input_shape.
        example_input=lambda: example_batch().as_args(),
        # Models train once, so "folds" are seeded evaluation bootstraps that give
        # the paired permutation test its variance. 10 -> 2^10 arrangements, far
        # finer p-value resolution than 5 folds (2^5=32, min p=0.031).
        cv_folds=CV_FOLDS,
        sanity_mutation=True,
        mutator_fraction=0.5,
        p_alpha=P_ALPHA,
        greater_is_better=True,  # F1: higher is better
        practical_margin=PRACTICAL_MARGIN,
        target_power=TARGET_POWER,
        seed=0,
    )

    print("verification ladder:")
    for gate in result.report.results:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name:16s} {gate.summary}")
    fire = result.report.get("fire_drill")
    if fire is not None:
        pvals = ", ".join(f"{s}={p:.3f}" for s, p in fire.data["p_degrade"].items())
        print(f"           slice sensitivity p_degrade: {pvals}")
    print()

    df = result.summary
    if df.empty:
        print(f"ladder halted at '{result.report.halted_at}'; no decisions produced")
    else:
        print("does each component earn its place? (mean F1 across folds)")
        header = (f"  {'ablation':20s} {'slice':15s} {'full_F1':>7s} {'abl_F1':>7s}"
                  f" {'p_deg':>6s} {'p_impr':>6s} {'MDE':>6s} {'power':>6s}  verdict")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in df.iterrows():
            print(
                f"  {row['variant']:20s} {row['slice']:15s} "
                f"{row['baseline_mean']:7.3f} {row['variant_mean']:7.3f} "
                f"{row['p_degrade']:6.3f} {row['p_improve']:6.3f} "
                f"{row['mde']:6.3f} {row['power_at_margin']:6.2f}  {_verdict(row)}"
            )
        print(
            f"\n  verdict key (alpha = {P_ALPHA}), both tails of the paired permutation"
            " test, plus power:\n"
            "    DEGRADED      removing the component significantly hurt   (p_deg  < alpha)\n"
            "    IMPROVED      removing it significantly helped            (p_impr < alpha)\n"
            f"    same          no significant difference, and the test COULD have seen a"
            f" {PRACTICAL_MARGIN:.3f} drop\n"
            "    INCONCLUSIVE  no significant difference, but the test was too weak to"
            " tell — do not\n"
            "                  read this as equivalence; add folds or shrink the slice's"
            " noise\n"
            f"\n  MDE   = smallest F1 drop this slice could detect at {TARGET_POWER:.0%}"
            " power (lower is better)\n"
            f"  power = probability of catching a {PRACTICAL_MARGIN:.3f} F1 drop, the"
            " margin declared as meaningful"
        )
        print("\ncomponent verdicts — REJECTED means the ablation degraded a slice,"
              "\ni.e. the component is load-bearing and must be kept:")
        for variant, decision in result.report.get("non_inferiority").data["decisions"].items():
            keep = "KEEP component" if decision == "REJECTED" else "component not justified"
            print(f"  => {variant:22s} {decision:9s} ({keep})")

    print("\ntraining time per variant (s): " +
          ", ".join(f"{a}={t:.1f}" for a, t in timings.items()))
    print(f"total wall time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
