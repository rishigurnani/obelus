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
        baseline_model=nn.Identity(),
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


def test_preflight_gate_does_not_mutate_baseline():
    import torch
    import torch.nn as nn

    from obelus.core.gates import PreflightGate
    from obelus.core.ladder import LadderContext

    model = nn.Linear(8, 8)
    before = model.weight.detach().clone()
    ctx = LadderContext(
        baseline_model=model,
        model_factory=lambda o: nn.Linear(8, 8),
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
