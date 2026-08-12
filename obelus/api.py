"""High-level automation API.

``run_ablation`` is the engine: it assembles the ladder context, runs the
:class:`~obelus.core.ladder.VerificationLadder`, and returns a rich
:class:`AblationResult`. ``AutoAblate`` is the thin spec-facing convenience
wrapper that returns just the summary DataFrame (with the full report tucked in
``df.attrs``), so a future CLI or reporter consumes the engine, not the wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from omegaconf import DictConfig

from obelus.adapters import HydraModelFactory
from obelus.core.discovery import generate_knockouts
from obelus.core.gates import FireDrillGate, default_gates
from obelus.core.ladder import (
    Gate,
    LadderContext,
    LadderReport,
    VerificationLadder,
)
from obelus.core.seams import ModelFactory, Scorer
from obelus.logging.tracker import NullTracker, Tracker

__all__ = ["AblationResult", "run_ablation", "AutoAblate"]

_SUMMARY_COLUMNS = [
    "variant",
    "slice",
    "p_degrade",
    "is_non_inferior",
    "p_improve",
    "is_improved",
    "baseline_mean",
    "variant_mean",
    "decision",
]


@dataclass(frozen=True)
class AblationResult:
    """Full result of an ablation run: the ladder report and a tidy summary."""

    report: LadderReport
    summary: pd.DataFrame

    @property
    def passed(self) -> bool:
        """Whether every gate ran and passed."""
        return self.report.passed

    @property
    def retained(self) -> list[str]:
        """Variant names that were RETAINED (empty if the ladder halted early)."""
        result = self.report.get("non_inferiority")
        if result is None:
            return []
        return [v for v, d in result.data["decisions"].items() if d == "RETAINED"]


def _summary_frame(report: LadderReport) -> pd.DataFrame:
    result = report.get("non_inferiority")
    rows = result.data["rows"] if result is not None else []
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def run_ablation(
    cfg: Optional[DictConfig] = None,
    *,
    scorer: Scorer,
    slices: list[str],
    model_factory: Optional[ModelFactory] = None,
    variants: Optional[dict[str, dict[str, Any]]] = None,
    input_shape: Optional[tuple[int, ...]] = None,
    example_input: Optional[Callable[[], Any]] = None,
    cv_folds: int = 5,
    sanity_mutation: bool = True,
    p_alpha: float = 0.05,
    greater_is_better: bool = True,
    mutator_fraction: float = 0.30,
    tracker: Optional[Tracker] = None,
    seed: int = 0,
    gates: Optional[list[Gate]] = None,
) -> AblationResult:
    """Run the 5-tier verification ladder over a model and its knockout variants.

    This is the engine behind :func:`AutoAblate`. It assembles a
    :class:`~obelus.core.ladder.LadderContext`, runs a
    :class:`~obelus.core.ladder.VerificationLadder`, and returns a rich
    :class:`AblationResult` (the ladder report plus a tidy summary frame).
    Prefer this over :func:`AutoAblate` for programmatic use — it exposes the
    full report, so a caller (or a future CLI/reporter) never has to reach into
    ``DataFrame.attrs``.

    The ladder (see :func:`obelus.core.gates.default_gates`) runs these gates in
    order and **halts on the first failure**, so expensive work never starts
    behind a broken guarantee:

    1. ``contracts``       — a baseline forward pass fires the layer's
       ``jaxtyping``/``beartype`` shape and ``icontract`` invariant checks.
    2. ``preflight``       — ``torchtest`` confirms every parameter moves under a
       gradient step, and ``hypothesis`` fuzzes finite inputs for NaN/Inf.
    3. ``fire_drill``      — an in-memory corrupted copy of the baseline must be
       flagged as degraded on *every* slice, proving the slices are sensitive
       (skipped when ``sanity_mutation=False``).
    4. ``cv_sweep``        — baseline vs. each knockout variant is scored across
       all ``(slice, fold)`` cells and logged to ``tracker``.
    5. ``non_inferiority`` — a one-tailed paired permutation test decides each
       variant RETAINED (non-inferior on every slice) or REJECTED (a
       statistically significant degradation on at least one slice).

    Gates 1–3 validate the *baseline* architecture and the benchmark itself;
    gates 4–5 are the actual ablation analysis. A REJECTED variant is a normal
    analytical outcome, **not** a ladder failure — the ladder only "fails" when a
    gate detects a pipeline/verification problem (a contract violation, a dead
    gradient, an insensitive slice, or an evaluation error).

    Construction modes
    ------------------
    The baseline model, the knockout variants, and the model factory can be
    supplied two ways:

    * **Hydra path** — pass ``cfg``. ``model_factory`` defaults to a
      :class:`~obelus.adapters.HydraModelFactory` (real ``hydra.utils.instantiate``)
      and ``variants`` defaults to :func:`~obelus.core.discovery.generate_knockouts`
      applied to ``cfg``.
    * **Direct/offline path** — pass ``model_factory`` and/or ``variants``
      explicitly. Anything passed directly overrides what would be derived from
      ``cfg``; ``cfg`` may then be omitted entirely (the offline/testing path).

    A ``"baseline"`` entry (empty override set) is always injected into
    ``variants`` if absent, and the baseline model is built once via
    ``model_factory(variants["baseline"])`` and shared across gates 1–3.

    Parameters
    ----------
    cfg:
        Hydra model config. Required only for whichever of ``model_factory`` /
        ``variants`` you do not pass directly; ignored for those you do.
    scorer:
        Callable ``(model, slice_name, fold) -> float`` that evaluates a model
        on one CV cell. **Required.** It owns all data loading and evaluation —
        obelus never touches your training scripts. Higher is better unless
        ``greater_is_better=False``. Must be deterministic given its inputs for
        reproducible results.
    slices:
        Failure-mode slice names (e.g. ``["val_full", "slice_long_seq"]``).
        **Required.** Passed verbatim to ``scorer`` as ``slice_name``.
    model_factory:
        Callable ``overrides -> nn.Module`` building a model from a knockout
        override set. Defaults to a ``HydraModelFactory`` over ``cfg``.
    variants:
        Mapping of variant name to its flat ``{dotted.path: value}`` override
        set. Defaults to knockouts discovered from ``cfg``. ``"baseline"`` is
        added automatically if missing.
    input_shape:
        Full input shape *including the batch dimension*, e.g. ``(8, 16, 512)``.
        Drives the dummy tensors for gates 1–2. Ignored when ``example_input`` is
        given; when both are ``None`` those two gates are reported as
        skipped-but-passed rather than run.
    example_input:
        Zero-arg callable returning a tensor, or a tuple of tensors, to feed
        gates 1–2. Use this for **multi-input models** (a molecular net taking
        graph tensors plus token ids, an encoder/decoder pair) whose forward
        signature ``input_shape`` cannot express. During fuzzing the first
        floating-point tensor is perturbed and the rest are held fixed.
    cv_folds:
        Number of cross-validation folds (``scorer`` is called with
        ``fold`` in ``range(cv_folds)``). Needs at least 2 for the permutation
        test. Default 5.
    sanity_mutation:
        When ``True`` (default) run Gate 3's benchmark fire drill; when ``False``
        drop it from the ladder.
    p_alpha:
        Significance threshold. A variant is non-inferior on a slice when
        ``p_degrade >= p_alpha``. Also used by the fire drill (a slice is
        sensitive when ``p_degrade < p_alpha``). Default 0.05.
    greater_is_better:
        Metric direction of ``scorer``. ``True`` for accuracy-like metrics;
        ``False`` for loss-like metrics (the comparison is flipped so the same
        engine handles both). Default ``True``.
    mutator_fraction:
        Fraction of each multi-dimensional weight tensor the fire drill zeroes
        when corrupting the baseline. Default 0.30.
    tracker:
        Experiment logger (:class:`~obelus.logging.tracker.Tracker`). Defaults to
        a no-op :class:`~obelus.logging.tracker.NullTracker`; pass an
        :class:`~obelus.logging.mlflow_local.MlflowTracker` for air-gapped
        SQLite MLflow logging.
    seed:
        Seed for the fire-drill corruption and every permutation test, making the
        whole run reproducible. Default 0.
    gates:
        Advanced escape hatch: an explicit ordered gate list, bypassing
        :func:`~obelus.core.gates.default_gates`. ``sanity_mutation=False`` still
        strips any :class:`~obelus.core.gates.FireDrillGate` from whatever list
        is used.

    Returns
    -------
    AblationResult
        ``.report`` is the ordered :class:`~obelus.core.ladder.LadderReport`
        (with ``.passed`` and ``.halted_at``); ``.summary`` is a tidy DataFrame
        with one row per ``(variant, slice)`` and columns ``variant``, ``slice``,
        ``p_degrade``, ``is_non_inferior``, ``baseline_mean``, ``variant_mean``,
        ``decision``. The summary is **empty** when the ladder halts before Gate
        5; inspect ``.report`` to see which gate stopped it and why.

    Raises
    ------
    ValueError
        If neither ``cfg`` nor ``model_factory`` is provided, or neither ``cfg``
        nor ``variants`` is provided.

    Notes
    -----
    Contract violations, dead gradients, NaN/Inf, or insensitive slices surface
    as a **failed gate** (``result.passed is False``) that halts the ladder — not
    as an exception — so the returned ``AblationResult`` is always well-formed.
    Genuinely exceptional conditions (a broken ``scorer`` or ``model_factory``)
    propagate as exceptions.

    Examples
    --------
    Offline, with explicit seams::

        result = run_ablation(
            scorer=my_eval_fn,                       # (model, slice, fold) -> float
            model_factory=my_factory,                # overrides -> nn.Module
            variants={"no_attn": {"model.attn._target_": "torch.nn.Identity"}},
            slices=["val_full", "slice_long_seq"],
            input_shape=(8, 16, 512),
            cv_folds=5,
        )
        if result.passed:
            print(result.retained)                   # variants safe to keep
        else:
            print("halted at", result.report.halted_at)

    Hydra path::

        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/model/transformer.yaml")
        result = run_ablation(cfg, scorer=my_eval_fn, slices=["val_full"],
                              input_shape=(8, 16, 512))
    """
    if model_factory is None:
        if cfg is None:
            raise ValueError("provide either cfg or model_factory")
        model_factory = HydraModelFactory(cfg)
    if variants is None:
        if cfg is None:
            raise ValueError("provide either cfg or variants")
        variants = generate_knockouts(cfg)
    variants = {"baseline": {}, **variants}  # guarantee a baseline entry

    if gates is None:
        gates = default_gates()
    if not sanity_mutation:
        gates = [g for g in gates if not isinstance(g, FireDrillGate)]

    ctx = LadderContext(
        baseline_model=model_factory(variants["baseline"]),
        model_factory=model_factory,
        variants=variants,
        scorer=scorer,
        slices=slices,
        folds=cv_folds,
        input_shape=input_shape,
        example_input=example_input,
        alpha=p_alpha,
        greater_is_better=greater_is_better,
        mutator_fraction=mutator_fraction,
        tracker=tracker or NullTracker(),
        seed=seed,
    )
    report = VerificationLadder(gates).run(ctx)
    return AblationResult(report=report, summary=_summary_frame(report))


def AutoAblate(
    cfg: Optional[DictConfig] = None,
    *,
    scorer: Scorer,
    slices: list[str],
    cv_folds: int = 5,
    sanity_mutation: bool = True,
    p_alpha: float = 0.05,
    db_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Spec-facing convenience wrapper: run an ablation, return the summary.

    ``db_path`` (if given) routes logging to a local SQLite MLflow store.
    ``output_dir`` (if given) receives a ``summary.csv`` snapshot. The full
    :class:`~obelus.core.ladder.LadderReport` is available as
    ``summary_df.attrs["ladder_report"]``.
    """
    tracker: Optional[Tracker] = None
    if db_path is not None:
        from obelus.logging.mlflow_local import init_local_tracker

        tracker = init_local_tracker(db_path)

    result = run_ablation(
        cfg,
        scorer=scorer,
        slices=slices,
        cv_folds=cv_folds,
        sanity_mutation=sanity_mutation,
        p_alpha=p_alpha,
        tracker=tracker,
        **kwargs,
    )
    summary = result.summary
    summary.attrs["ladder_report"] = result.report

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "summary.csv", index=False)

    return summary
