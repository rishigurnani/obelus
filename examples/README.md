# Example: config-driven architecture ablation on BACE-1

A runnable, offline demonstration of obelus on a real drug-discovery task —
classifying molecules as **active / inactive against BACE-1** (human β-secretase 1,
an Alzheimer's target). It answers the question obelus exists for:

> Does every component of my architecture **earn its place** — or is one of them
> dead weight, and is another quietly load-bearing on a failure mode the
> aggregate metric never shows me?

Runs on a CPU MacBook Air in **~32 s**.

```bash
uv pip install --python .venv -e ".[examples]"   # rdkit + scikit-learn
.venv/bin/python -m examples.molnet_ablation.run
```

## The point: the ablation matrix comes from the Hydra config

You write **one YAML** ([`configs/architecture.yaml`](configs/architecture.yaml)),
and `obelus.core.discovery.generate_knockouts` derives the entire experiment.
No variant list is written by hand:

```yaml
model:
  _target_: examples.molnet_ablation.layers.HybridMolecularClassifier
  descriptor_branch:  { _target_: ...DescriptorBranch, ... }
  graph_encoder:      { _target_: ...GraphEncoder, residual: { active: true } }
  sequence_encoder:   { _target_: ...SequenceEncoder, ... }
  fusion:             { _target_: ...GatedFusion, ... }
```

becomes five knockouts, discovered automatically:

| Variant | Override generated | Architectural question |
|---|---|---|
| `no_descriptor_branch` | `_target_` → `torch.nn.Identity` | do hand-crafted 2D features still matter? |
| `no_graph_encoder` | `_target_` → `torch.nn.Identity` | does the GNN earn its cost? |
| `no_sequence_encoder` | `_target_` → `torch.nn.Identity` | does the Transformer add anything? |
| `no_fusion` | `_target_` → `torch.nn.Identity` | is learned gating better than a plain mean? |
| `no_residual` | `active` → `false` | do the GNN skip connections matter? |

`torch.nn.LayerNorm`-style **library** layers are deliberately left alone —
discovery only ablates modules matching your project prefix. **Add a module to
the YAML and it gets ablated on the next run**; delete one and its variant
disappears. That is the leverage Hydra provides here.

Hydra does the instantiation for real: `HydraModelFactory` applies each override
set and calls `hydra.utils.instantiate`, so a knockout is a genuine config-level
architecture change, not a special code path.

## The architecture (mixed hand-crafted and learned features)

```
descriptors ──► DescriptorBranch   (2D computed RDKit features)  ┐
atom graph  ──► GraphEncoder       (learned — graph convolution) ├─► Fusion ─► head
SMILES ids  ──► SequenceEncoder    (learned — Transformer)       ┘
```

A branch knocked out to `nn.Identity` is treated as *absent*, so fusion simply
receives one fewer embedding ([`layers.py`](molnet_ablation/layers.py)).
`GatedFusion.forward` carries obelus's `@check_shapes` / `@check_invariants`
contracts, so Gate 1 does real work on this model.

Because the model takes six tensors, gates 1–2 are fed a real `example_input`
batch rather than a single `input_shape`.

## Failure modes → slices (one split per failure mode)

Each variant trains **once** on a shared pool, then is probed on disjoint cohorts
held out from it, each stressing a documented way a molecular classifier breaks
([`data.py`](molnet_ablation/data.py)):

| Slice | Failure mode it exposes |
|---|---|
| `in_distribution` | control — random held-out molecules |
| `novel_scaffold` | structural novelty (Murcko scaffolds unseen in training) |
| `large` | molecular-size extrapolation (top-decile heavy atoms) |
| `lipophilic` | physicochemical covariate shift (top-decile logP) |
| `flexible` | conformational flexibility (top-decile rotatable bonds) |

## How folds work here

Models train once, so the CV `folds` are **seeded bootstrap resamples** of each
slice — they give the paired permutation test its variance, and baseline and
variant always see the same resample per fold. The example uses 10 (2¹⁰
arrangements → fine p-value resolution). **F1 is the guiding metric.**

## Reading the output

A variant here is an *ablation*, so the verdicts inverts in a useful way:

You must declare `EFFECT_SIZE` — the move that actually counts (0.10 F1 here).
obelus requires it; there is no default, because "is this difference real?" is
unanswerable until you say what size of difference matters.

A variant must **earn its place**: RETAINED requires a `FORWARD` leap on at
least one slice *and* no `BACKWARD` slice anywhere. Being merely harmless is not
enough — `ON_PAR` everywhere is REJECTED.

- **RETAINED** — removing the component measurably *helps* somewhere and hurts
  nowhere ⇒ the removal earns its place.
- **REJECTED** — either some slice went BACKWARD (the component is
  **load-bearing, keep it**) or nothing improved (no reason to change).

Per row, **both tails** of the same paired permutation test are reported, so
every label is a significance claim rather than a mean comparison — and the
*null* result is qualified by power:

| Label | Condition | Meaning |
|---|---|---|
| `BACKWARD` | significantly worse by ≥ δ | a real regression — rejects the ablation |
| `FORWARD` | significantly better by ≥ δ | a real gain |
| `ON_PAR` | neither leap established | the move is smaller than δ, or not significant |
| `*` suffix | `ON_PAR` but underpowered | the test could not have seen δ — *absence of evidence* |

Both conditions must hold: a difference counts only when it is **large enough to
matter** *and* **statistically significant**. A 0.003 drift never labels, and
neither does a statistically clean 0.03 drop when δ = 0.10.

Testing against δ is not new machinery: it is the same paired permutation test
with the baseline shifted by ±δ, so `classify_effect` just calls
`evaluate_non_inferiority` twice.

## Is each test adequately powered?

A non-inferiority gate that RETAINS a variant because it *couldn't detect
anything* is the "insensitive test battery → false confidence" failure the spec
names as Problem #2. Two power columns make that visible:

Power is a *curve* over effect size. The two columns read that one curve from
opposite ends, which is why their headers name the value each is evaluated at:

- **`MDE@80%`** — fix power at 80%, solve for the effect: the smallest F1 drop
  the slice can detect. Lower is better; a large MDE means it is too noisy to trust.
- **`pwr@0.100`** — fix the effect at the declared `EFFECT_SIZE`, solve for
  power. Below `TARGET_POWER` (0.8), an `ON_PAR` row is starred (`*`) because it
  cannot be read as equivalence.

The same δ drives both the labels and the power, so they cannot drift apart:
`pwr@0.100 >= 0.8` exactly when `MDE@80% <= 0.100`.

**Neither is evaluated at the observed difference.** A row showing a 0.006 drop
with `MDE@80% = 0.013` and `pwr@0.100 = 1.00` is consistent, not contradictory —
0.100 > 0.013, so its power necessarily exceeds 0.8. Power *at* the observed
effect is post-hoc power: a monotone restatement of the p-value that always calls
non-significant results underpowered, so obelus never reports it.

The distinction is not academic. Two `ON_PAR` rows can mean opposite things — one
is evidence the component is dispensable, the other only evidence that the slice
is too noisy to say.

Power here is closed-form (noncentral *t*), so it costs ~4 ms for all 25 cells
rather than the thousands of resamples a Monte-Carlo estimate would need.
Post-hoc "observed power" is deliberately not reported — it is just a
restatement of the p-value. See [`obelus/core/power.py`](../obelus/core/power.py).

### The structural check

Before any data is involved, the run prints what the test is even *capable* of:

```
test capability: 10 folds -> smallest attainable p = 0.0010 vs alpha = 0.05  [OK]
  (5 folds would give 0.0312; 4 folds 0.0625 > 0.05, i.e. structurally unable to reject)
```

A paired sign-flip permutation test over `n` folds has only `2**n` arrangements,
so p can never fall below `2**-n`. At `alpha=0.05` a **4-fold** setup can never
reject — every slice would pass vacuously — and the spec's default of 5 folds
clears the bar only by requiring the single most extreme arrangement.

Exact numbers shift with platform math libraries, but the shape is stable and
genuinely interesting. The descriptor branch and the learned fusion are broadly
load-bearing. The graph encoder is the striking one: dropping it *significantly
helps* on `large` and `lipophilic` while *significantly hurting* on
`novel_scaffold` and `flexible` — a real trade-off that a single aggregate
validation score would average into nothing.
