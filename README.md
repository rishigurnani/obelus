# obelus

Named after the Alexandrian manuscript mark (÷) used to flag doubtful or
interpolated text, **obelus** subjects deep-learning models to trial-by-fire
validation: dynamic contracts, pre-flight fuzzing, benchmark-sensitivity checks,
5-fold cross-validation, and a non-parametric non-inferiority gate — all offline.

See `spec/` for the product and technical specs. This repository implements
**Phase 1 (Core Engine)**.

## The 5-tier verification ladder

Execution halts immediately when a gate fails, saving compute:

1. **Contracts & invariants** — `jaxtyping`/`beartype` shapes, `icontract` NaN/Inf/norm bounds.
2. **Pre-flight fuzzing & gradient integrity** — `hypothesis` + `torchtest`.
3. **Benchmark sensitivity fire drill** — in-memory sabotage proves slices detect corruption.
4. **5-fold cross-validation sweep** — baseline vs. knockout variants across failure-mode slices.
5. **Non-inferiority decision** — one-tailed paired permutation test, `p_degrade >= alpha` on every slice.

## Design

The five gates are `Gate` objects run by a single short-circuiting
`VerificationLadder`. Gate 3 and Gate 5 share one mechanism — evaluate two
models across folds×slices, then run the permutation test — differing only in
acceptance polarity (expect-degraded vs. expect-non-inferior).

Heavy/environment-specific concerns are injectable seams so the ladder runs
offline in tests:

- `ModelFactory` — builds a model from base config + knockout overrides.
- `Scorer` — scores a model on `(slice, fold)`; users supply their own eval fn.
- `Tracker` — experiment logging (`NullTracker`, or `MlflowTracker` behind the `mlflow` extra).

## Quick start

```python
from omegaconf import OmegaConf
from obelus import AutoAblate

cfg = OmegaConf.load("configs/model/transformer.yaml")
summary_df = AutoAblate(
    cfg=cfg,
    scorer=my_eval_fn,              # (model, slice, fold) -> float
    input_shape=(8, 16, 512),
    slices=["val_full", "slice_long_seq", "slice_noisy"],
    cv_folds=5,
    sanity_mutation=True,
    p_alpha=0.05,
    db_path="sqlite:///local_obelus.db",
)
print(summary_df)                  # summary_df.attrs["ladder_report"] holds the full run
```

## Example

[`examples/`](examples/README.md) gates three architectures — a 2D-descriptor MLP
baseline vs. a graph net and a SMILES transformer — on the real **BACE-1
active/inactive** dataset, across five failure-mode slices. Runs offline on a CPU
in ~30 s:

```bash
uv pip install --python .venv -e ".[examples]"
.venv/bin/python -m examples.molnet_ablation.run
```
