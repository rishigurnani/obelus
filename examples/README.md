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

- **REJECTED** — removing the component significantly hurt some slice ⇒ the
  component is **load-bearing, keep it**.
- **RETAINED** — the model was non-inferior without it ⇒ that component is not
  justified by this benchmark.

Per row, **both tails** of the same paired permutation test are reported, so
every label is a significance claim rather than a mean comparison:

| Label | Condition | Meaning |
|---|---|---|
| `DEGRADED` | `p_degrade < alpha` | removing the component significantly hurt |
| `IMPROVED` | `p_improve < alpha` | removing it significantly helped |
| `same` | neither | no significant difference in either direction |

A higher mean F1 is **not** enough to earn `IMPROVED` — a slice can drift up by
0.003 on pure noise, and that reads `same`.

Exact numbers shift with platform math libraries, but the shape is stable and
genuinely interesting. The descriptor branch and the learned fusion are broadly
load-bearing. The graph encoder is the striking one: dropping it *significantly
helps* on `large` and `lipophilic` while *significantly hurting* on
`novel_scaffold` and `flexible` — a real trade-off that a single aggregate
validation score would average into nothing.
