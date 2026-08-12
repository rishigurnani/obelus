# Example: gating molecular architectures on BACE-1

A runnable, offline demonstration of obelus on a real drug-discovery task —
classifying molecules as **active / inactive against BACE-1** (human β-secretase 1,
an Alzheimer's target). It answers the question obelus exists for:

> Are feature-*learning* models (a graph net, a SMILES transformer) **non-inferior**
> to a strong 2D-descriptor baseline across every known failure mode — or does one
> slip through the aggregate metric while quietly regressing on a hard slice?

Runs on a CPU MacBook Air in **~30 s** (three tiny models, trained once each).

```bash
uv pip install --python .venv -e ".[examples]"   # rdkit + scikit-learn
.venv/bin/python -m examples.molnet_ablation.run
```

## Data

`data/bace.csv` — 1513 molecules with SMILES and a binary `label` (active/inactive
against BACE-1), from the MoleculeNet BACE dataset, trimmed to two columns and
bundled so the example is fully offline.

## Three architectures (the obelus "variants")

The baseline is compared against two learned-representation variants:

| Model | Representation | How features arise |
|---|---|---|
| `descriptor_mlp` (baseline) | 16 RDKit **2D computed descriptors** → MLP | hand-computed |
| `gnn` | atom graph → 2-layer normalized graph conv | **learned** |
| `smiles_transformer` | SMILES characters → 2-layer Transformer encoder | **learned** |

Each is a plain `nn.Module` exposing `fit(smiles, y)` / `predict_proba(smiles)` and
owning its own featurizer, so obelus's scorer stays architecture-agnostic. See
[`models.py`](molnet_ablation/models.py).

## Failure modes → slices (one split per failure mode)

Each architecture trains **once** on a shared pool; we then probe it on disjoint
cohorts held out from that pool, each stressing a documented way a molecular
classifier breaks (see [`data.py`](molnet_ablation/data.py)):

| Slice | Failure mode it exposes |
|---|---|
| `in_distribution` | control — random held-out molecules |
| `novel_scaffold` | structural novelty (Murcko scaffolds unseen in training) |
| `large` | molecular-size extrapolation (top-decile heavy atoms) |
| `lipophilic` | physicochemical covariate shift (top-decile logP) |
| `flexible` | conformational flexibility (top-decile rotatable bonds) |

## How folds work here

Because each model trains once, the CV `folds` are **seeded bootstrap resamples**
of each slice — they give the paired permutation test its variance, and baseline
and variant always see the same resample per fold. The example uses 10 (2¹⁰
arrangements → fine p-value resolution); F1 is the guiding metric.

## What you'll see

The 5-tier ladder runs and, if the baseline and benchmark are sound, prints a
per-`(variant, slice)` table of baseline vs. variant F1 with `p_degrade` and a
RETAINED/REJECTED verdict per variant. A variant is **REJECTED** if it degrades
significantly on *even one* slice — so a model that looks fine on average but
regresses on a single failure mode is caught. Exact numbers depend on your
platform's math libraries, but the shape of the finding is stable: the
undertrained GNN is rejected across the board, while the transformer is
non-inferior on the hard OOD slices yet can be rejected for a single
in-distribution regression — precisely the zero-tolerance behavior obelus enforces.

For the one-call convenience form, swap `run_ablation` for `AutoAblate` (returns
just the summary DataFrame, with the full report in `df.attrs["ladder_report"]`).
