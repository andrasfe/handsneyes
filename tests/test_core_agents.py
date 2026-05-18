"""Tests for handsneyes.core.agents.{base,context}."""

from __future__ import annotations

import numpy as np

from handsneyes.core.agents.base import Outcome
from handsneyes.core.agents.context import AgentContext


def test_outcome_bool_semantics() -> None:
    assert bool(Outcome(success=True)) is True
    assert bool(Outcome(success=False)) is False


def test_outcome_default_fields() -> None:
    o = Outcome(success=True)
    assert o.reason == ""
    assert o.data == {}


def test_agent_context_constructs_with_no_args() -> None:
    ctx = AgentContext()
    assert ctx.mouse is None
    assert ctx.keyboard is None
    assert ctx.capture is None
    assert ctx.output_dir is None
    assert ctx.scratch == {}


def test_record_frame_no_output_dir_returns_none() -> None:
    ctx = AgentContext()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert ctx.record_frame(img, label="frame") is None
