"""The VerificationLadder runner: ordering and short-circuiting."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from obelus.core.ladder import (
    GateResult,
    LadderContext,
    VerificationLadder,
)


@dataclass
class RecordingGate:
    name: str
    passes: bool
    log: list

    def run(self, ctx, prior) -> GateResult:
        self.log.append(self.name)
        return GateResult(self.name, self.passes, "recorded")


def _ctx():
    return LadderContext(
        model_factory=lambda o: nn.Identity(),
        variants={"baseline": {}},
        scorer=lambda m, s, k: 0.0,
        slices=[],
    )


def test_all_gates_run_when_passing():
    log: list = []
    ladder = VerificationLadder(
        [RecordingGate("a", True, log), RecordingGate("b", True, log)]
    )
    report = ladder.run(_ctx())
    assert log == ["a", "b"]
    assert report.passed is True
    assert report.halted_at is None


def test_build_inputs_prefers_example_input_over_shape():
    import torch

    ctx = _ctx()
    assert ctx.build_inputs() is None  # neither supplied -> gates skip

    ctx.input_shape = (2, 8)
    single = ctx.build_inputs()
    assert len(single) == 1 and single[0].shape == (2, 8)

    ctx.example_input = lambda: (torch.zeros(3, 4), torch.ones(3, dtype=torch.bool))
    multi = ctx.build_inputs()
    assert len(multi) == 2 and multi[1].dtype == torch.bool


def test_preflight_gate_does_not_mutate_baseline():
    import torch
    import torch.nn as nn

    from obelus.core.gates import PreflightGate
    from obelus.core.ladder import LadderContext

    model = nn.Linear(8, 8)
    before = model.weight.detach().clone()
    ctx = LadderContext(
        model_factory=lambda o: model,
        variants={"baseline": {}},
        scorer=lambda m, s, k: 0.0,
        slices=[],
        input_shape=(4, 8),
    )
    result = PreflightGate().run(ctx, [])
    assert result.passed is True
    assert torch.equal(model.weight, before)  # clone was tested, not the original


def test_failure_short_circuits_later_gates():
    log: list = []
    ladder = VerificationLadder(
        [
            RecordingGate("a", True, log),
            RecordingGate("b", False, log),
            RecordingGate("c", True, log),
        ]
    )
    report = ladder.run(_ctx())
    assert log == ["a", "b"]  # "c" never ran
    assert report.passed is False
    assert report.halted_at == "b"
    assert report.get("c") is None


def test_fire_drill_diagnoses_why_a_slice_is_blind():
    """A failing fire drill must say what the sabotage cost, not just that it failed."""
    import pytest
    import torch.nn as nn

    from obelus.core.gates import FireDrillGate
    from obelus.core.ladder import LadderContext

    ctx = LadderContext(
        model_factory=lambda o: nn.Linear(4, 4),
        variants={"baseline": {}},
        scorer=lambda m, s, k: 0.5 + 0.001 * k,  # ignores the model: nothing to destroy
        slices=["numb"],
        folds=10,
    )
    result = FireDrillGate().run(ctx, [])
    assert result.passed is False
    # Nothing was destroyed because there was no signal: the message must name
    # that cause, not just report that the slice failed.
    assert "no signal here to destroy" in result.summary
    assert "numb" in result.summary
    assert result.data["sabotage_cost"]["numb"] == pytest.approx(0.0, abs=1e-9)
