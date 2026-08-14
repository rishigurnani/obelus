"""Decision policies — the one place the two directions differ.

Everything else in the ladder is shared. Ablation and insertion diverge only in
what counts as an acceptable outcome, and that follows from **who carries the
burden of proof**:

* **Ablation** proposes something *cheaper*. Parsimony gives it the benefit of
  the doubt, so it is accepted unless harm is shown. But because weak evidence
  *accepts* here, "no significant difference" is not good enough — a noisy slice
  would wave removals through. Ablation therefore demands proven equivalence
  (TOST) and treats an unproven slice as blocking.
* **Insertion** proposes something *more expensive*. The addition must earn its
  place with a real gain somewhere and no regression anywhere. Weak evidence
  declines, which already fails safe, so plain non-significance suffices.

Adding a third direction later is a new policy here and nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from obelus.core.options import COMPLEXIFY, SIMPLIFY
from obelus.core.stats import BACKWARD, FORWARD, INCONCLUSIVE

__all__ = ["DecisionPolicy", "AblationPolicy", "InsertionPolicy", "RETAINED", "REJECTED"]

RETAINED, REJECTED = "RETAINED", "REJECTED"


@runtime_checkable
class DecisionPolicy(Protocol):
    """How per-slice labels become a verdict on the variant as a whole."""

    name: str
    direction: str
    requires_equivalence: bool

    def decide(self, labels: list[str]) -> str: ...

    def explain(self, decision: str) -> str: ...


class AblationPolicy:
    """Accept a simplification unless it is shown to cost something."""

    name = "ablation"
    direction = SIMPLIFY
    requires_equivalence = True  # ON_PAR must be proven, not merely unrefuted

    def decide(self, labels: list[str]) -> str:
        # INCONCLUSIVE blocks: acceptance rests on equivalence, so a slice that
        # could not establish it is not evidence the change is safe.
        blocking = {BACKWARD, INCONCLUSIVE}
        return REJECTED if any(l in blocking for l in labels) else RETAINED

    def explain(self, decision: str) -> str:
        return (
            "the component is dispensable — drop it"
            if decision == RETAINED
            else "keep the component"
        )


class InsertionPolicy:
    """Accept a complication only if it demonstrably buys something."""

    name = "insertion"
    direction = COMPLEXIFY
    requires_equivalence = False  # declining on weak evidence already fails safe

    def decide(self, labels: list[str]) -> str:
        if BACKWARD in labels:
            return REJECTED
        return RETAINED if FORWARD in labels else REJECTED

    def explain(self, decision: str) -> str:
        return (
            "the addition earns its place — adopt it"
            if decision == RETAINED
            else "not worth the added complexity"
        )
