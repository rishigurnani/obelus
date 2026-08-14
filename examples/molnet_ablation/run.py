"""Architecture verification on BACE-1, in both directions, from one config.

`obelus` reads ``configs/architecture_options.yaml`` — where every decision node
lists its alternatives ordered simple -> complex and names the incumbent — and
derives two experiments from it:

    AutoInsert  (complexify)  is this addition worth its complexity?
                              accept only if a gain is PROVEN somewhere
    AutoAblate  (simplify)    can this component be dropped?
                              accept unless harm is proven — but equivalence
                              must be PROVEN, since acceptance rests on it

Same dataset, same slices, same ladder. Only the burden of proof moves, which is
why an identical row of evidence can be REJECTED in one direction and RETAINED in
the other.

Run:  python -m examples.molnet_ablation.run   (needs the ``examples`` extra)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import matthews_corrcoef

from examples.molnet_ablation.data import SLICE_DOC, build_dataset
from examples.molnet_ablation.features import MolFeaturizer, example_batch
from examples.molnet_ablation.training import fit, predict_proba
from obelus import run_verification
from obelus.adapters import HydraModelFactory
from obelus.core.options import generate_moves
from obelus.core.policy import AblationPolicy, InsertionPolicy
from obelus.core.power import min_achievable_p

_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CONFIG_PATH = _CONFIGS / "architecture_options.yaml"

P_ALPHA = 0.05
CV_FOLDS = 30
TARGET_POWER = 0.8
# The two directions ask different questions, so they take different thresholds:
# giving up a little to simplify is a fair trade, while paying for extra
# complexity should demand more in return.
TOLERANCE = 0.03  # AutoAblate: MCC I will give up to get something simpler
WORTHWHILE_GAIN = 0.05  # AutoInsert: MCC an addition must buy to justify itself

def _make_factory(cfg, moves, train_batch, train_y, timings, seed=0):
    """Hydra builds the architecture; this wrapper trains what Hydra returns."""
    hydra_factory = HydraModelFactory(cfg)
    # Reverse-map an override set back to its move name, for readable timings.
    # Values may be whole config nodes (dicts), so key on a stable repr.
    def _key(ov):
        return repr(sorted(((k, repr(v)) for k, v in ov.items())))

    names = {_key(ov): name for name, ov in moves.items()}

    def factory(overrides):
        started = time.perf_counter()
        # Seed before instantiation, not just before training: weight init draws
        # from the global RNG, and the earlier gates consume a variable amount of
        # it (hypothesis fuzzing). Without this, every variant would be compared
        # against a differently-initialised baseline and runs would not repeat.
        torch.manual_seed(seed)
        model = hydra_factory(overrides)  # real hydra.utils.instantiate
        fit(model, train_batch, train_y, seed=seed)
        timings[names.get(_key(overrides), "baseline")] = time.perf_counter() - started
        return model

    return factory


def _make_scorer(dataset, featurizer):
    """MCC on a seeded bootstrap of a slice, one resample per fold.

    MCC, not F1: F1's trivial baseline moves with prevalence (predicting "active"
    everywhere scores 0.79 on a 66%-positive cohort), so a corrupted model that
    collapses to a constant *gains* on the skewed slices and the fire drill sees
    no damage. MCC scores any constant predictor 0 on every slice, so the same
    corruption is measured against a common zero.

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
        return matthews_corrcoef(labels[boot], (probs[boot] >= 0.5).astype(int))

    return scorer


def _render(df, effect_size: float) -> None:
    """One table renderer for both directions.

    Column headers name the effect size each power number is evaluated AT: the
    two are inverse readings of one power curve, and an unqualified "power"
    invites reading it as power at the observed difference (it is not).
    """
    header = (f"    {'move':22s} {'slice':16s} {'base':>6s} {'var':>6s}"
              f" {'p_back*':>7s} {'p_fwd*':>7s}"
              f" {'MDE@' + format(TARGET_POWER, '.0%'):>8s}"
              f" {'pwr@' + format(effect_size, '.3f'):>9s}  verdict")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for _, row in df.iterrows():
        print(f"    {row['variant']:22s} {row['slice']:16s} "
              f"{row['baseline_mean']:6.3f} {row['variant_mean']:6.3f} "
              f"{row['p_backward']:7.3f} {row['p_forward']:7.3f} "
              f"{row['mde']:8.3f} {row['power_at_effect']:9.2f}  {row['label']}")


def _run_direction(policy, effect_size, cfg, dataset, featurizer, train_batch, train_y):
    moves = generate_moves(cfg, policy.direction)
    if len(moves) < 2:
        print(f"    (no {policy.direction} moves available in this config)\n")
        return

    print(f"    moves: {', '.join(k for k in moves if k != 'baseline')}")
    timings: dict[str, float] = {}
    result = run_verification(
        cfg,
        policy=policy,
        scorer=_make_scorer(dataset, featurizer),
        model_factory=_make_factory(cfg, moves, train_batch, train_y, timings),
        variants=moves,
        # The membership, not just the names: these cohorts overlap, so obelus
        # applies a Holm correction to the "any slice rejects" family.
        slices=dataset.slices,
        # The model takes six tensors, so gates 1-2 need a real example batch
        # rather than a single input_shape.
        example_input=lambda: example_batch().as_args(),
        # Models train once, so "folds" are seeded evaluation bootstraps that give
        # the paired permutation test its variance. 30 rather than 10 because the
        # `large` cohort resolves only ~0.076 MCC at 10, while the fire drill's
        # sabotage costs it 0.047 — real damage the slice could not see. Note what
        # extra folds do buy: they shrink *resampling* noise on this fixed cohort,
        # sharpening the baseline-vs-variant comparison; they add no new molecules
        # and so say nothing more about the population.
        cv_folds=CV_FOLDS,
        sanity_mutation=True,
        mutator_fraction=0.5,
        p_alpha=P_ALPHA,
        greater_is_better=True,  # MCC: higher is better
        effect_size=effect_size,
        target_power=TARGET_POWER,
        seed=0,
    )

    for gate in result.report.results:
        print(f"    [{'PASS' if gate.passed else 'FAIL'}] {gate.name:12s} {gate.summary}")
    df = result.summary
    if df.empty:
        print(f"    halted at '{result.report.halted_at}'\n")
        return

    print()
    _render(df, effect_size)
    print()
    for variant, decision in result.report.get("decision").data["decisions"].items():
        print(f"    => {variant:22s} {decision:9s} ({policy.explain(decision)})")
    print("    training time (s): "
          + ", ".join(f"{a}={t:.1f}" for a, t in timings.items()) + "\n")


def main() -> None:
    start = time.perf_counter()
    cfg = OmegaConf.load(CONFIG_PATH)

    dataset = build_dataset(seed=0)
    train_smiles, train_y = dataset.subset(dataset.train_idx)
    featurizer = MolFeaturizer().fit(train_smiles)
    train_batch = featurizer.transform(train_smiles)

    print("BACE-1 architecture verification — one config, both directions\n")
    print(f"config: {CONFIG_PATH.relative_to(Path.cwd())}")
    print(f"training molecules: {len(dataset.train_idx)}")
    print("failure-mode slices  (scored by MCC: any constant predictor scores 0):")
    for name, idx in dataset.slices.items():
        shared = {o: len(np.intersect1d(idx, dataset.slices[o]))
                  for o in dataset.slices if o != name}
        overlap = ", ".join(f"{o}:{k}" for o, k in shared.items() if k)
        print(f"  {name:16s} n={len(idx):4d}  prevalence="
              f"{float(dataset.labels[idx].mean()):.2f}"
              f"  shares: {overlap or 'nothing'}")
    print("  Cohorts overlap, so each measures its own failure mode rather than"
          " whatever an\n  earlier cohort left over. Shared molecules make the"
          " per-slice tests dependent, so\n  the 'any slice rejects' family takes"
          " a Holm correction; the fire drill needs none\n  (it requires *every*"
          " slice, which is already level-alpha).")

    # Structural check, independent of any data: with n paired folds the test has
    # only 2**n sign-flip arrangements, so p can never fall below 2**-n. If that
    # floor exceeds alpha, every slice would pass — vacuously.
    floor = min_achievable_p(CV_FOLDS)
    print(f"\ntest capability: {CV_FOLDS} folds -> smallest attainable p = {floor:.4f}"
          f" vs alpha = {P_ALPHA}  [{'OK' if floor <= P_ALPHA else 'IMPOSSIBLE'}]")
    # Holm divides alpha by the slice count for the most significant slice, so the
    # fold budget must clear the *corrected* bar, not the nominal alpha.
    strictest = P_ALPHA / len(dataset.slices)
    print(f"  Holm's strictest rung is alpha/{len(dataset.slices)} = {strictest:.4f}"
          f"  [{'OK' if floor <= strictest else 'IMPOSSIBLE'}]"
          f" — 5 folds ({min_achievable_p(5):.4f}) could not clear it\n")

    print(f"[1] AutoInsert — is the addition worth its complexity?"
          f"  needs +{WORTHWHILE_GAIN} MCC somewhere, no regression anywhere")
    _run_direction(InsertionPolicy(), WORTHWHILE_GAIN, cfg, dataset, featurizer,
                   train_batch, train_y)

    print(f"[2] AutoAblate — can the component be dropped?"
          f"  tolerates -{TOLERANCE} MCC, but equivalence must be PROVEN")
    _run_direction(AblationPolicy(), TOLERANCE, cfg, dataset, featurizer,
                   train_batch, train_y)

    print("verdict key — labels are judged against that direction's effect size:")
    print("  BACKWARD      significantly WORSE by >= delta   -> rejects either direction")
    print("  FORWARD       significantly BETTER by >= delta  -> required by AutoInsert")
    print("  ON_PAR        neither leap; for AutoAblate, equivalence was PROVEN (TOST)")
    print("  INCONCLUSIVE  equivalence unproven — blocks an ablation, since accepting")
    print("                would rest on evidence the slice never produced")
    print(f"  MDE@{TARGET_POWER:.0%}       smallest MCC move the slice can detect"
          f" (fix power, solve for effect)")
    print("  pwr@delta     probability of catching a delta-sized move"
          " (fix effect, solve for power)")
    print("  p_back*/p_fwd*  family-wise adjusted across slices (see gate summary)")
    print(f"\ntotal wall time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
