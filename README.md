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
4. **Cross-validation sweep** — baseline vs. each variant across data set slices.
5. **Decision** — each slice classified FORWARD / ON_PAR / BACKWARD against a
   declared effect size, then the active policy's rule applied.

Gates 1–2 run on **every variant**, not just the baseline: an insertion contains
code no baseline has, and even a removal can destabilise numerics the baseline
never saw.

Gate 5 uses **both tails** of the paired permutation test, so an improvement is a
significance claim rather than a higher mean, and a difference must be both large
enough to matter (≥ the declared effect size) and statistically significant. It
also reports each slice's **power** — the minimum detectable effect, and whether
the slice could have caught a move you called meaningful — so a null result from
an underpowered slice is never mistaken for equivalence.

## Design

The five gates are `Gate` objects run by a single short-circuiting
`VerificationLadder`. Gate 3 and Gate 5 share one mechanism — evaluate two
models across folds×slices, then run the permutation test — differing only in
acceptance polarity. Testing against an effect-size margin is that same test with
the baseline shifted by ±δ, and the TOST equivalence bounds are the two tails it
would otherwise discard: one primitive, four questions.

Heavy/environment-specific concerns are injectable seams so the ladder runs
offline in tests:

- `ModelFactory` — builds a model from base config + move overrides.
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
    tolerance=0.02,                # the F1 loss you would accept to simplify
    db_path="sqlite:///local_obelus.db",
)
print(summary_df)                  # summary_df.attrs["ladder_report"] holds the full run
```

## Two directions

Architectural change comes in two flavours, and they carry opposite burdens of
proof. One config expresses both — each decision node lists its alternatives
ordered simple → complex, and names the incumbent:

```yaml
graph_encoder:
  _options_:
    - {_target_: torch.nn.Identity}          # rank 0 — simplest
    - {_target_: src.layers.GraphEncoder}    # rank 1
  _current_: 1
```

| | question | RETAINED when |
|---|---|---|
| `AutoAblate` | can this be removed or simplified? | no slice regresses by `tolerance`, **and** equivalence is proven on every slice |
| `AutoInsert` | is this addition worth its complexity? | it gains `worthwhile_gain` on at least one slice and regresses on none |

Ablation accepts on "nothing was lost", so weak evidence would wave changes
through — it therefore demands proven equivalence (TOST) and blocks on a slice
too noisy to establish it. Insertion declines on weak evidence, which already
fails safe. The difference lives entirely in
[`obelus/core/policy.py`](obelus/core/policy.py); the ladder is shared.

## Example

[`examples/`](examples/README.md) runs **both** directions on the real **BACE-1
active/inactive** dataset. One Hydra YAML declares a molecular architecture
(2D-descriptor branch + GNN + SMILES Transformer + learned fusion) with each
decision node's alternatives ordered simple → complex; obelus derives the
additions and the removals from it and gates them across a few data set
slices. The payoff is a pair of moves with *identical* evidence and opposite
verdicts. Runs offline on a CPU in ~12 s:

```bash
uv pip install --python .venv -e ".[examples]"
.venv/bin/python -m examples.molnet_ablation.run
```
