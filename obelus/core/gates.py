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
from obelus.core.power import analyze_power
from obelus.core.runner import run_preflight
from obelus.core.stats import (
    BACKWARD,
    FORWARD,
    INCONCLUSIVE,
    ON_PAR,
    adjust_pvalues,
    classify_effect,
    evaluate_non_inferiority,
    label_effect,
    select_correction,
)

__all__ = [
    "ContractGate",
    "PreflightGate",
    "FireDrillGate",
    "CrossValidationSweepGate",
    "DecisionGate",
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
        models = ctx.models()
        for name, model in models.items():
            try:
                model.eval()
                with torch.no_grad():
                    model(*inputs)
            except Exception as exc:
                return GateResult(self.name, False, f"contract violation in '{name}': {exc}")
        return GateResult(self.name, True, f"contracts satisfied on {len(models)} model(s)")


class PreflightGate:
    """Gate 2 — gradient integrity + numerical-stability fuzzing on the baseline."""

    name = "preflight"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        inputs = ctx.build_inputs()
        if inputs is None:
            return GateResult(self.name, True, _SKIP_NO_SHAPE)
        models = ctx.models()
        for name, model in models.items():
            # run_preflight takes an optimizer step, so test a throwaway clone
            # rather than perturb a model reused downstream.
            try:
                run_preflight(copy.deepcopy(model), example_inputs=inputs)
            except Exception as exc:
                return GateResult(self.name, False, f"pre-flight failed for '{name}': {exc}")
        return GateResult(
            self.name, True,
            f"gradients flow, no NaN/Inf from finite inputs, across {len(models)} model(s)")


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
        # What the sabotage actually cost, per slice — the key diagnostic.
        cost = {v.slice_name: v.test.baseline_mean - v.test.variant_mean for v in verdicts}
        data = {"p_degrade": p_values, "sabotage_cost": cost}
        # Sensitive == degradation detected == NOT judged non-inferior.
        blind = [v for v in verdicts if v.test.is_non_inferior]
        if not blind:
            return GateResult(
                self.name, True, f"all {len(verdicts)} slice(s) detected the corruption", data
            )
        # Name the cause, not just the numbers: the two failure modes need
        # opposite fixes, and the caller cannot tell them apart from a p-value.
        lines = []
        for v in blind:
            mde = analyze_power(v.baseline, v.variant, alpha=ctx.alpha,
                                greater_is_better=ctx.greater_is_better).mde
            drop = cost[v.slice_name]
            if drop < mde:  # the damage is real but smaller than the slice can see
                why = (f"breaking the model cost {drop:.3f}, but this slice cannot "
                       f"resolve anything below {mde:.3f} — too noisy to judge; "
                       f"add folds or enlarge the cohort")
            else:  # nothing was destroyed, because there was nothing there
                why = (f"breaking the model cost only {drop:.3f} — there is no signal "
                       f"here to destroy, so this slice cannot show it would catch "
                       f"a regression; fix the model on this slice or drop it")
            lines.append(f"\n      {v.slice_name}: {why}")
        return GateResult(
            self.name,
            False,
            f"{len(blind)} of {len(verdicts)} slice(s) cannot detect deliberate "
            f"sabotage, so they cannot be trusted to catch a real regression:"
            + "".join(lines),
            data,
        )


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
                    variant_model = ctx.models()[variant_name]
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


class DecisionGate:
    """Gate 5 — decide RETAINED/REJECTED per variant, per the active policy.

    Each slice is classified FORWARD / ON_PAR / BACKWARD relative to
    ``ctx.effect_size``. A variant is RETAINED iff it is **FORWARD on at least
    one slice and BACKWARD on none** — it must earn its place with a real gain,
    not merely avoid harm, so a variant that is ON_PAR everywhere is rejected.
    Sub-threshold wobble neither earns nor rejects: a move counts only when it is
    both large enough to matter and statistically significant.
    """

    name = "decision"

    def run(self, ctx: LadderContext, prior: list[GateResult]) -> GateResult:
        sweep = next((r.data.get("sweep") for r in prior if r.name == "cv_sweep"), None)
        if sweep is None:
            return GateResult(self.name, False, "no CV sweep results available")

        rows: list[dict] = []
        decisions: dict[str, str] = {}
        # One family per variant: each variant is its own accept/reject decision,
        # so strictness does not drift as the config gains unrelated variants.
        method = select_correction(ctx.slice_members)
        for variant_name, pairs in sweep.items():
            equivalence: dict[str, bool] = {}
            variant_rows: list[dict] = []
            for pair in pairs:
                effect = classify_effect(
                    pair.baseline,
                    pair.variant,
                    ctx.effect_size,
                    alpha=ctx.alpha,
                    greater_is_better=ctx.greater_is_better,
                    require_equivalence=ctx.policy.requires_equivalence,
                    seed=ctx.seed,
                )
                # The same declared delta drives power, so ON_PAR can be read as
                # evidence of absence rather than absence of evidence.
                power = analyze_power(
                    pair.baseline,
                    pair.variant,
                    alpha=ctx.alpha,
                    target_power=ctx.target_power,
                    margin=ctx.effect_size,
                    greater_is_better=ctx.greater_is_better,
                )
                equivalence[pair.slice_name] = effect.is_equivalent
                variant_rows.append(
                    {
                        "variant": variant_name,
                        "slice": pair.slice_name,
                        "p_backward": effect.p_backward,
                        "p_forward": effect.p_forward,
                        "baseline_mean": effect.baseline_mean,
                        "variant_mean": effect.variant_mean,
                        "mde": power.mde,
                        "power_at_effect": power.power_at_margin,
                        "adequately_powered": power.adequately_powered,
                        "p_not_worse": effect.p_not_worse,
                        "p_not_better": effect.p_not_better,
                    }
                )
            # "Any slice rejects" is union-intersection, so these two carry the
            # family-wise penalty; the TOST bounds behind is_equivalent need
            # *every* slice and so take none. Columns hold the adjusted values,
            # because those are what the verdict was actually read from.
            for key in ("p_backward", "p_forward"):
                adjusted = adjust_pvalues({r["slice"]: r[key] for r in variant_rows}, method)
                for row in variant_rows:
                    row[key] = adjusted[row["slice"]]
            for row in variant_rows:
                row["label"] = label_effect(
                    row["p_backward"], row["p_forward"], equivalence[row["slice"]],
                    alpha=ctx.alpha, require_equivalence=ctx.policy.requires_equivalence,
                )
                # ON_PAR the test could never have contradicted is not evidence;
                # flag it so it is never read as equivalence.
                row["inconclusive"] = row["label"] == INCONCLUSIVE or (
                    row["label"] == ON_PAR and not row["adequately_powered"]
                )
            labels = [row["label"] for row in variant_rows]
            # The one place the two directions differ (see obelus.core.policy).
            decision = ctx.policy.decide(labels)
            decisions[variant_name] = decision
            for row in variant_rows:
                row["decision"] = decision
            rows.extend(variant_rows)

        n_retained = sum(d == "RETAINED" for d in decisions.values())
        summary = (f"{n_retained}/{len(decisions)} variant(s) retained"
                   f"; {len(ctx.slices)} slices, {method} correction")
        n_inconclusive = sum(r["inconclusive"] for r in rows)
        if n_inconclusive:
            summary += f"; {n_inconclusive}/{len(rows)} cell(s) underpowered"
        return GateResult(
            self.name,
            True,
            summary,
            {
                "rows": rows,
                "decisions": decisions,
                "n_inconclusive": n_inconclusive,
            },
        )


def default_gates() -> list:
    """The canonical 5-tier ladder, in order."""
    return [
        ContractGate(),
        PreflightGate(),
        FireDrillGate(),
        CrossValidationSweepGate(),
        DecisionGate(),
    ]
