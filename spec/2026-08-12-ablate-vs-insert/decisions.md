# Resolutions to the open questions in `spec.md`

## Q1 — How does `AutoInsert` learn what to insert? → **one config, ordered alternatives per node**

Adopted, with three refinements to the proposal.

Each decision node declares its alternatives **ordered simple → complex**, plus
which one is currently in use:

```yaml
model:
  attn:
    _options_:                                   # ordered: simple -> complex
      - {_target_: torch.nn.Identity}            # rank 0
      - {_target_: src.layers.LinearAttention}   # rank 1
      - {_target_: src.layers.FullAttention, dim: 512}  # rank 2
    _current_: 2                                 # the incumbent
```

From one config obelus derives **both** directions, with no second file and no
guessing:

- `AutoAblate` variants = every option ranked **below** `_current_`
- `AutoInsert` variants = every option ranked **above** `_current_`

**Refinement 1 — ordering replaces the binary flag.** The proposal was "two
options; if one is Identity the other is the complex one, else the user says
which is simpler". Making `_options_` an *ordered list* subsumes both cases: the
order *is* the user's declaration of what simpler means, so nothing extra is
needed when neither is Identity, and it generalises past two options
(Identity / linear / full attention is a natural three). `torch.nn.Identity` is
then only a convenience: obelus warns if it appears anywhere but rank 0.

**Refinement 2 — the incumbent must be explicit.** The proposal states the
alternatives but not which is currently in force, and obelus cannot build the
baseline without it. Hence `_current_`. It also makes the direction meaningful:
the same node yields ablation candidates or insertion candidates purely
depending on where the incumbent sits.

**Refinement 3 — the existing conventions become special cases.** A node with
`active: true` is exactly `_options_: [absent, present], _current_: 1`, and
today's `_target_ → nn.Identity` knockout is the rank-0 move. `generate_knockouts`
stays as the shorthand path; the ordered form is the general one.

**Known limitation (accepted).** Complexity is assumed to be a *total* order per
node. Two genuinely incomparable alternatives (a GNN vs. a Transformer of similar
cost) cannot be ranked honestly. That is a **swap**, not an ablation or an
insertion, and it takes the insertion rule — you should not churn architecture
for a wash. No new policy needed; it is `InsertionPolicy` applied to a
same-rank move, deferred until asked for.

---

## Q2 — One δ or two? → **two, named per direction**

```python
AutoAblate(..., tolerance=0.02)        # how much I will give up to get simpler
AutoInsert(..., worthwhile_gain=0.05)  # how much it must buy to justify the cost
```

Both feed the same `classify_effect` δ and the same power analysis; only the
names and the defaults-you-would-choose differ.

---

## Q3 — Correct for multiplicity on the FORWARD family? → **No. The margin already does it.**

The concern in `spec.md` D5 was **wrong** and that section is corrected.

The stated worry was that "FORWARD on at least one of S slices" is a union over S
tests, inflating false adoptions to `1 − (1−α)^S` ≈ 23% at S = 5. That figure
assumes a **zero-margin superiority test**. `p_forward` does not test against the
baseline; it tests against `baseline + δ`. A variant with no real effect sits a
full δ below that boundary and essentially never clears it.

Simulated, 800 runs, 10 folds, S = 5, slices powered so MDE ≈ δ:

| variant's true effect | per-slice P(FORWARD) | P(accept over 5 slices) |
|---|---|---|
| identical to baseline (global null) | **0.0000** | **0.0000** |
| exactly +δ | 0.0495 | 0.2275 |

The 23% only appears once the variant **genuinely delivers the declared
meaningful gain**, where adopting it is the correct outcome — that is power, not
error. Under the null the observed false-adoption rate is zero.

Residual nuance, accepted without machinery: a variant whose true effect is just
under δ on many slices becomes easier to adopt as S grows. It is still delivering
most of the gain you called meaningful, so this is a benign boundary effect, not
a false adoption.

Note also what the "no BACKWARD anywhere" clause is really for. Under the null it
adds little (P(BACKWARD) is also ~0). Its job is **mixed** variants — ones that
help on one slice and hurt on another — which is exactly the aggregate-metric
blind spot the project exists to catch.

---

## Q4 — Does ablation's `ON_PAR` require TOST equivalence? → **Yes**

`AutoAblate` accepts *on* `ON_PAR`, so failure-to-reject is too weak a basis: a
noisy slice would silently approve removals. Ablation therefore requires
**positive proof of equivalence** — both one-sided tests must reject, showing the
effect lies within ±tolerance.

`AutoInsert` keeps plain non-significance, because there `ON_PAR` is the
*non-adopting* condition and weak evidence already fails safe.

Both use tails already computed by the two shifted tests in `classify_effect`,
so this is a reading of existing numbers, not new statistics. Consequence: a
slice too noisy to prove equivalence blocks an ablation instead of rubber-
stamping it, which is the intended conservative direction. The existing
`inconclusive` flag becomes the *reason* for such a rejection rather than an
advisory asterisk.
