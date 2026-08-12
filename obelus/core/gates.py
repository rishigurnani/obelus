"""The five concrete gates of the verification ladder.

Each gate is small and self-contained; they share the evaluation/stats
primitives in :mod:`obelus.core.evaluator` and :mod:`obelus.core.stats` rather
than re-implementing them. Gate 3 and Gate 5 are the same paired-comparison
mechanism with opposite acceptance polarity.
"""

from __future__ import annotations

import copy

import torch

from obelus.core.evaluator import compare_across_slices, evaluate_pair
from obelus.core.ladder import GateResult, LadderContext
from obelus.core.mutator import ModelMutator
from obelus.core.runner import run_preflight
from obelus.core.stats import evaluate_non_inferiority

__all__ = [
    "ContractGate",
    "PreflightGate",
    "FireDrillGate",
    "CrossValidationSweepGate",
    "NonInferiorityGate",
    "default_gates",
]

_SKIP_NO_SHAPE = "skipped: no input_shape provided"


class ContractGate:
    """Gate 1 — trigger the baseline's shape/invariant contracts via a forward pass."""

    name = "contracts"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        inputs = ctx.build_inputs()
        if inputs is None:
            return GateResult(self.name, True, _SKIP_NO_SHAPE)
        try:
            ctx.baseline_model.eval()
            with torch.no_grad():
                ctx.baseline_model(*inputs)
        except Exception as exc:
            return GateResult(self.name, False, f"contract violation: {exc}")
        return GateResult(self.name, True, "contracts satisfied on baseline forward pass")


class PreflightGate:
    """Gate 2 — gradient integrity + numerical-stability fuzzing on the baseline."""

    name = "preflight"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        inputs = ctx.build_inputs()
        if inputs is None:
            return GateResult(self.name, True, _SKIP_NO_SHAPE)
        # run_preflight takes an optimizer step, so test a throwaway clone rather
        # than perturb the (possibly already-trained) baseline shared downstream.
        probe = copy.deepcopy(ctx.baseline_model)
        try:
            run_preflight(probe, example_inputs=inputs)
        except Exception as exc:
            return GateResult(self.name, False, f"pre-flight failed: {exc}")
        return GateResult(self.name, True, "gradients flow; no NaN/Inf from finite inputs")


class FireDrillGate:
    """Gate 3 — prove each slice detects a known corruption of the baseline.

    A slice is *sensitive* when the corrupted model is flagged as degraded on it
    (``p_degrade < alpha``). The gate fails if any slice is blind to the sabotage.
    """

    name = "fire_drill"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        gen = torch.Generator().manual_seed(ctx.seed)
        corrupted = ModelMutator.create_corrupted_variant(
            ctx.baseline_model, ctx.mutator_fraction, generator=gen
        )
        verdicts = compare_across_slices(
            ctx.baseline_model,
            corrupted,
            ctx.slices,
            ctx.folds,
            ctx.scorer,
            alpha=ctx.alpha,
            greater_is_better=ctx.greater_is_better,
            seed=ctx.seed,
        )
        p_values = {v.slice_name: v.test.p_degrade for v in verdicts}
        # Sensitive == degradation detected == NOT judged non-inferior.
        insensitive = [v.slice_name for v in verdicts if v.test.is_non_inferior]
        passed = not insensitive
        if passed:
            summary = f"all {len(verdicts)} slice(s) detected the corruption"
        else:
            summary = f"insensitive slice(s): {', '.join(insensitive)}"
        return GateResult(self.name, passed, summary, {"p_degrade": p_values})


class CrossValidationSweepGate:
    """Gate 4 — evaluate baseline vs. every knockout variant across folds×slices."""

    name = "cv_sweep"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        sweep: dict[str, list] = {}
        try:
            for variant_name, overrides in ctx.variants.items():
                if variant_name == "baseline":
                    continue
                with ctx.tracker.run(variant_name):
                    ctx.tracker.log_params(
                        {f"override.{k}": v for k, v in overrides.items()}
                    )
                    variant_model = ctx.model_factory(overrides)
                    pairs = evaluate_pair(
                        ctx.baseline_model,
                        variant_model,
                        ctx.slices,
                        ctx.folds,
                        ctx.scorer,
                    )
                    sweep[variant_name] = pairs
                    for pair in pairs:
                        mean_v = sum(pair.variant) / len(pair.variant)
                        ctx.tracker.log_metric(f"{pair.slice_name}.variant_mean", mean_v)
        except Exception as exc:
            return GateResult(self.name, False, f"evaluation failed: {exc}")
        n_variants = len(sweep)
        return GateResult(
            self.name,
            True,
            f"evaluated {n_variants} variant(s) across {len(ctx.slices)} slice(s)",
            {"sweep": sweep},
        )


class NonInferiorityGate:
    """Gate 5 — decide RETAINED/REJECTED per variant via the permutation test.

    A variant is RETAINED iff it is non-inferior on *every* slice; a single
    significant degradation (``p_degrade < alpha``) rejects it.
    """

    name = "non_inferiority"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        sweep = next((r.data.get("sweep") for r in prior if r.name == "cv_sweep"), None)
        if sweep is None:
            return GateResult(self.name, False, "no CV sweep results available")

        rows: list[dict] = []
        decisions: dict[str, str] = {}
        for variant_name, pairs in sweep.items():
            non_inferior_all = True
            variant_rows: list[dict] = []
            for pair in pairs:
                test = evaluate_non_inferiority(
                    pair.baseline,
                    pair.variant,
                    alpha=ctx.alpha,
                    greater_is_better=ctx.greater_is_better,
                    seed=ctx.seed,
                )
                non_inferior_all &= test.is_non_inferior
                variant_rows.append(
                    {
                        "variant": variant_name,
                        "slice": pair.slice_name,
                        "p_degrade": test.p_degrade,
                        "is_non_inferior": test.is_non_inferior,
                        "p_improve": test.p_improve,
                        "is_improved": test.is_improved,
                        "baseline_mean": test.baseline_mean,
                        "variant_mean": test.variant_mean,
                    }
                )
            decision = "RETAINED" if non_inferior_all else "REJECTED"
            decisions[variant_name] = decision
            for row in variant_rows:
                row["decision"] = decision
            rows.extend(variant_rows)

        n_retained = sum(d == "RETAINED" for d in decisions.values())
        summary = f"{n_retained}/{len(decisions)} variant(s) retained"
        return GateResult(
            self.name, True, summary, {"rows": rows, "decisions": decisions}
        )


def default_gates() -> list:
    """The canonical 5-tier ladder, in order."""
    return [
        ContractGate(),
        PreflightGate(),
        FireDrillGate(),
        CrossValidationSweepGate(),
        NonInferiorityGate(),
    ]
