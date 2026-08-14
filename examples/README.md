# Example: architecture verification on BACE-1, in both directions

A runnable, offline demo on a real drug-discovery task — classifying molecules as
**active / inactive against BACE-1** (human β-secretase 1, an Alzheimer's
target). It answers the two questions obelus exists for:

> Is this component worth **adding**? Can this component be **dropped**?

Trains 6 small models end to end in well under a minute on a CPU laptop.

```bash
uv pip install --python .venv -e ".[examples]"   # rdkit + scikit-learn
.venv/bin/python -m examples.molnet_ablation.run
```

## One config, both directions

[`configs/architecture_options.yaml`](configs/architecture_options.yaml) declares
each decision node's alternatives **ordered simple → complex**, and names the
incumbent:

```yaml
graph_encoder:
  _options_:
    - {_target_: torch.nn.Identity}                              # rank 0
    - {_target_: examples.molnet_ablation.layers.GraphEncoder}   # rank 1
  _current_: 1        # present today -> an ablation candidate
```

The ordering *is* your declaration of what "simpler" means, so obelus never
guesses. From that one file it derives both experiments — nothing is hand-written:

| direction | options it uses | question |
|---|---|---|
| `AutoInsert` | ranked **above** `_current_` | is it worth adding? |
| `AutoAblate` | ranked **below** `_current_` | can it be dropped? |

The model sits deliberately mid-stack — descriptor branch and GNN on, SMILES
Transformer and learned fusion off — so there are moves in both directions. Hydra
instantiates for real, so a move is a genuine config-level architecture change,
not a special code path. This is obelus's only variant form: a knockout is just
the rank-0 option, and a toggle is a two-option node over the flag.

## The architecture

```
descriptors ──► DescriptorBranch   (2D computed RDKit features)  ┐
atom graph  ──► GraphEncoder       (learned — graph convolution) ├─► Fusion ─► head
SMILES ids  ──► SequenceEncoder    (learned — Transformer)       ┘
```

A branch at rank 0 (`nn.Identity`) counts as *absent*, so fusion receives one
fewer embedding ([`layers.py`](molnet_ablation/layers.py)). `GatedFusion.forward`
carries obelus's `@check_shapes` / `@check_invariants` contracts, so Gate 1 does
real work. The model takes six tensors, so gates 1–2 get a real `example_input`
batch rather than an `input_shape` — and run on **every** variant.

## Failure modes → slices

Each variant trains **once** on a shared pool, then is probed on cohorts held out
from it ([`data.py`](molnet_ablation/data.py)): `in_distribution` (control),
`novel_scaffold`, `large`, `flexible`, `lipophilic`.

Three design choices there are worth knowing, because each fixes a specific way
this benchmark can lie to you:

- **Cohorts overlap.** Big molecules are also flexible. Forcing each molecule
  into one cohort would silently redefine `flexible` as "flexible *but small*",
  and the slice would stop measuring the failure mode it is named after.
- **Overlap ⇒ dependent tests**, so the "any slice rejects" family takes a Holm
  correction. Passing slice *membership* (not just names) is what lets obelus
  detect this; see `obelus.core.stats.select_correction` for why Holm rather than
  a permutation-based correction.
- **Cohorts keep half their tail in training** (`tail_support`). Held out
  entirely, the model never sees a single large molecule, and where label
  prevalence flips across the cut it learns the relationship backwards — scoring
  *negative* MCC on its own cohort.

**MCC, not F1.** F1's trivial baseline moves with prevalence: predicting "active"
everywhere scores 0.79 on a 66%-positive cohort. A corrupted model collapses
toward a constant and so *gains* on skewed slices, which blinds the fire drill.
Any constant predictor scores 0 MCC on every slice.

Folds are seeded bootstrap resamples of each slice (models train once) giving the
paired permutation test its variance; baseline and variant always see the same
resample per fold.

## Reading the output

Each `(move, slice)` cell is labelled against that direction's effect size δ:

| Label | Meaning |
|---|---|
| `BACKWARD` | significantly worse by ≥ δ — rejects the move in either direction |
| `FORWARD` | significantly better by ≥ δ — required by `AutoInsert` |
| `ON_PAR` | neither leap; for `AutoAblate`, equivalence was **proven** (TOST) |
| `INCONCLUSIVE` | equivalence unproven — blocks an ablation |

```
AutoInsert  RETAIN iff  >=1 FORWARD  and  no BACKWARD     (must earn its cost)
AutoAblate  RETAIN iff  no BACKWARD  and  no INCONCLUSIVE (harm ruled out, provably)
```

δ differs by direction on purpose: `TOLERANCE` is what you will give up to
simplify, `WORTHWHILE_GAIN` what an addition must buy. **The same row of evidence
can therefore be RETAINED in one direction and REJECTED in the other** — an
addition that buys nothing is not worth its cost, while a removal that costs
nothing is free simplification. That asymmetry is the whole point, and it lives
entirely in [`obelus/core/policy.py`](../obelus/core/policy.py).

`MDE@80%` and `pwr@δ` are inverse readings of one power curve: fix power and
solve for effect, or fix effect and solve for power. They cannot disagree —
`pwr@δ ≥ 0.8` exactly when `MDE@80% ≤ δ`. Neither is evaluated at the *observed*
difference, which would be post-hoc power: a monotone restatement of the p-value.
This matters most for `AutoAblate`, which accepts *on* `ON_PAR` — so a slice too
noisy to prove equivalence is reported `INCONCLUSIVE` and blocks the change
rather than waving it through. See [`power.py`](../obelus/core/power.py).

If Gate 3 fails, the run halts before any sweep and names the cause per slice —
either the model has no signal there to destroy, or the cohort is too noisy to
resolve the damage. Those need opposite fixes, which is why the message
distinguishes them. Refusing to sweep behind a benchmark that cannot detect
regressions is the tool working, not a bug.
