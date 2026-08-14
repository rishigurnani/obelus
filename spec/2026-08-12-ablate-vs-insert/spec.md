---
author: obelus-core-team
created: 2026-08-12
status: implemented
supersedes: partial — extends 2026-08-10-initial Gate 5
---

# obelus — Ablation vs. Insertion: two directions of architectural change

## Summary

`obelus` currently supports one direction of change: **remove a component and
prove nothing broke**. That is the wrong test for the opposite move — **add a
component and prove it was worth it**. This spec separates the two into
`AutoAblate` and `AutoInsert`, which share the entire verification ladder and
differ only in a **`DecisionPolicy`** and a **variant generator**.

The user chooses which to call. `obelus` never infers whether a change is a
simplification or a complication — "simpler" is a judgement about cost,
maintenance, and intent that only the caller can make.

---

## The core asymmetry: who carries the burden of proof

Every other difference follows from this one.

|  | `AutoAblate` (simplify) | `AutoInsert` (complicate) |
|---|---|---|
| Variant is | **cheaper** than baseline | **more expensive** than baseline |
| Default when evidence is weak | **adopt the variant** (parsimony) | **keep the baseline** (status quo) |
| Burden of proof is on | keeping the complexity | the addition |
| Dangerous error | accepting a removal that really did hurt (**Type II** on degradation) | accepting an addition that does not really help (**Type I** on improvement) |
| Therefore the binding constraint is | **statistical power** (weak evidence *accepts*, so it must be proven) | **effect size** (weak evidence *declines*, which already fails safe) |

That last row is the crux. It is why only ablation needs TOST equivalence (D6)
and why only ablation treats low power as a blocking condition (D8) — and,
conversely, why insertion needs no multiplicity correction (D5).

### Acceptance rules

Let each `(variant, slice)` cell be classified `FORWARD` / `ON_PAR` / `BACKWARD`
against a declared effect size δ (existing `classify_effect`).

```
AutoAblate   RETAIN  iff  no slice is BACKWARD
                          (ON_PAR everywhere is sufficient — the win is the simplification)

AutoInsert   RETAIN  iff  at least one slice is FORWARD  AND  no slice is BACKWARD
                          (must earn its place; ON_PAR everywhere buys nothing but cost)
```

---

## The differences

**D1 — Acceptance rule.** As above. This is the only difference the user named;
the rest are consequences.

**D2 — Direction of the null hypothesis.** Ablation asks "can I rule out harm?".
Insertion asks "can I demonstrate benefit?". Same two-tailed machinery, opposite
tail carries the decision. Already supported: `classify_effect` returns both
`p_backward` and `p_forward`.

**D3 — Variant generation.** `generate_knockouts` only knows how to *subtract*
(`_target_ → nn.Identity`, `active → false`). Insertion needs the inverse
generator, and there is no way to invent a module the config does not mention.
*Resolved: one config, ordered `_options_` per node — see `decisions.md`.*

**D4 — Which model gates 1–2 verify → per-variant, in both directions.**

**Insertion variants contain code the baseline does not.** The new module is
simultaneously the only thing under test, the least-exercised code in the run,
and the one thing baseline verification structurally cannot reach. Demonstrated:
a new adapter with a dead parameter (a learnable gate declared but never wired
into `forward`) passes gates 1–2 against the baseline — it must, since the
baseline has no such parameter — and fails immediately against the variant
(`inner.2.gate did not change`).

The damage is not a crash. The ladder proceeds, trains the variant, and returns a
well-formed `ON_PAR` → REJECTED verdict. The conclusion "this module does not
help" is false; half of it was never connected. A silent wrong answer, at full
GPU cost.

*Correction to an earlier draft of this section:* it claimed that for ablation,
verifying the baseline "covers every variant". That holds for **code** — no new
module executes, so contract/shape bugs in unseen modules are impossible — but
not for **numerics**: removing a normalisation layer can send activations to Inf
in a variant that was stable in the baseline.

Therefore gates 1–2 run **per variant in both directions**: mandatory for
insertion, still valuable for ablation, and cheap either way (a forward pass plus
~25 fuzz examples). This removes D4 as a policy difference — `DecisionPolicy`
needs no `verify_variants()` member.

*Implementation note:* variants are currently constructed inside Gate 4, so
per-variant verification means either building them once up front or folding the
checks into the sweep. Prefer the former — it keeps the short-circuit property
(a broken variant is caught before any training).

**D5 — Multiplicity: not a problem here.** *(This section previously claimed a
~23% false-adoption rate and proposed Holm correction. That was wrong; see
`decisions.md` Q3 for the simulation.)*

Insertion's rule is a **union** ("at least one FORWARD"), which would inflate
false positives over S slices — but only for a *zero-margin* superiority test.
`p_forward` tests against `baseline + δ`, so a variant with no real effect sits a
full δ from the boundary and essentially never clears it: measured
P(FORWARD | no effect) = 0.0000, P(accept over 5 slices) = 0.0000. **The margin
supplies the protection a correction would have.** No adjustment is applied.

Ablation's rule is an **intersection** (every slice non-BACKWARD), which is
self-correcting on acceptance for the usual reason.

**D6 — Evidentiary standard for `ON_PAR`.** `ON_PAR` currently means "neither
leap was established" — a *failure to reject*, not evidence of equivalence.

- For **insertion**, `ON_PAR` is the *rejecting* condition, so weak evidence
  fails safe: no adoption. Plain non-significance is fine.
- For **ablation**, `ON_PAR` is the *accepting* condition. Accepting on a
  failure to reject means a noisy slice silently approves removals — the exact
  "false confidence" failure the ladder exists to prevent. The rigorous form is
  **TOST equivalence**: prove the effect lies within ±δ, which requires *two*
  one-sided tests to both reject. Both are already computable from the existing
  shifted tests (the two tails we currently discard).
  *Resolved: yes — TOST for ablation only.*

**D7 — Meaning of δ.** The two directions use δ to answer different questions,
and a sane user would set them to different numbers:

- Ablation: δ is a **tolerance** — "how much am I willing to lose to get
  something simpler?" (often small, e.g. 0.02 F1)
- Insertion: δ is a **worthwhile gain** — "how much must it buy me to justify
  the cost?" (often larger, e.g. 0.05 F1)

*Resolved: two parameters, `tolerance` and `worthwhile_gain`.*

**D8 — Role of power.** Follows from D6. For ablation, the power/MDE columns are
a *gating* concern (an underpowered `ON_PAR` must not silently accept). For
insertion they are *diagnostic* (an underpowered run just fails to adopt, and
tells you to add folds). Same computation, different consequence.

**D9 — Reporting.** `decision` keeps one meaning in the engine — "adopt the
variant" — which is what makes the two policies share everything. Only the
human-facing gloss differs: for ablation RETAINED reads *"the component is
dispensable"*; for insertion RETAINED reads *"the component earns its place"*.

**D10 — Composition (noted, out of scope).** Components that are individually
`ON_PAR` to remove may jointly degrade; modules that individually help may not
compose. Both directions need a "verify the composed winner" pass. Deferred.

---

## Architecture

Zero churn to the ladder. Gates 1–4 are untouched. The only new abstraction:

```python
# obelus/core/policy.py
class DecisionPolicy(Protocol):
    name: str
    def decide(self, labels: list[str]) -> str: ...  # D1: RETAINED / REJECTED

class AblationPolicy:    # no BACKWARD
class InsertionPolicy:   # >=1 FORWARD and no BACKWARD
```

`NonInferiorityGate` (rename → `DecisionGate`) delegates its accept rule to
`ctx.policy` instead of hard-coding it — a one-line change at the single site
that currently computes `any_forward and no_backward`.

`AutoAblate` and `AutoInsert` become thin wrappers over the existing
`run_ablation` engine (rename → `run_verification`), differing only in the
policy and variant generator they pass. Adding a third direction later (e.g.
"replace-in-place / swap") is a new `DecisionPolicy`, touching nothing else.

### API sketch

```python
AutoAblate(cfg, scorer=..., slices=[...], tolerance=0.02, ...)   # policy=AblationPolicy
AutoInsert(cfg, scorer=..., slices=[...], worthwhile_gain=0.05, ...)  # policy=InsertionPolicy
```

---

## Non-goals

- Deciding *for* the user whether a change is a simplification. The caller picks
  the function; obelus does not measure "complexity".
- Automatic composition of multiple accepted changes (D10).
- Any change to Gates 1–4 mechanics beyond D4 (verifying each variant, not
  only the baseline).

---

## Resolved

All four open questions are answered in `decisions.md`:

1. **Insertion source** — one config; each decision node lists `_options_`
   ordered simple→complex plus `_current_`. Both directions derive from it.
2. **δ** — two per-direction parameters: `tolerance` / `worthwhile_gain`.
3. **Multiplicity** — no correction; the δ margin already provides it (D5 above
   corrected).
4. **`ON_PAR`** — ablation requires TOST equivalence; insertion keeps plain
   non-significance.
