"""The decision policies — the only place the two directions differ."""

from __future__ import annotations

import pytest

from obelus.core.policy import AblationPolicy, InsertionPolicy

ABLATE, INSERT = AblationPolicy(), InsertionPolicy()


@pytest.mark.parametrize(
    "labels, ablation, insertion",
    [
        # Simplification is accepted on equivalence; an addition must earn its keep.
        (["ON_PAR", "ON_PAR"], "RETAINED", "REJECTED"),
        # A real gain with no regression satisfies both.
        (["FORWARD", "ON_PAR"], "RETAINED", "RETAINED"),
        # One regression rejects either direction.
        (["FORWARD", "BACKWARD"], "REJECTED", "REJECTED"),
        (["ON_PAR", "BACKWARD"], "REJECTED", "REJECTED"),
        # Unproven equivalence blocks an ablation (it accepts ON_PAR) but is
        # irrelevant to an insertion, which declines on anything short of a gain.
        (["INCONCLUSIVE", "ON_PAR"], "REJECTED", "REJECTED"),
        (["INCONCLUSIVE", "FORWARD"], "REJECTED", "RETAINED"),
    ],
)
def test_decision_matrix(labels, ablation, insertion):
    assert ABLATE.decide(labels) == ablation
    assert INSERT.decide(labels) == insertion


def test_directions_and_equivalence_requirements():
    assert (ABLATE.direction, ABLATE.requires_equivalence) == ("simplify", True)
    assert (INSERT.direction, INSERT.requires_equivalence) == ("complexify", False)
