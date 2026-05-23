"""Tests for the PlanExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from handsneyes.core.agents.context import AgentContext
from handsneyes.core.agents.controller import PlanStep
from handsneyes.core.agents.executor import PlanExecutor, StepResult


class TestPlanExecutor:
    @pytest.mark.asyncio
    async def test_empty_plan(self) -> None:
        ex = PlanExecutor(AgentContext())
        results = await ex.run([])
        assert results == []

    @pytest.mark.asyncio
    async def test_unknown_agent_fails_clean(self) -> None:
        ex = PlanExecutor(AgentContext())
        results = await ex.run([
            PlanStep(agent="bogus", kwargs={}, rationale="testing"),
        ])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, StepResult)
        assert not r.outcome.success
        assert "unknown agent" in r.outcome.reason

    @pytest.mark.asyncio
    async def test_key_combo_step(self) -> None:
        kb = AsyncMock()
        ex = PlanExecutor(AgentContext(keyboard=kb))
        results = await ex.run([
            PlanStep(
                agent="key_combo",
                kwargs={"modifiers": ["super"], "key": "l"},
                rationale="lock",
            ),
        ])
        assert len(results) == 1
        r = results[0]
        assert r.outcome.success
        kb.send_key_combo.assert_awaited_once_with(["super"], "l")

    @pytest.mark.asyncio
    async def test_scroll_step(self) -> None:
        mouse = AsyncMock()
        ex = PlanExecutor(AgentContext(mouse=mouse))
        results = await ex.run([
            PlanStep(
                agent="scroll",
                kwargs={
                    "direction": "down",
                    "amount": 3,
                    "between_ticks": 0.0,
                    "post_settle": 0.0,
                },
                rationale="scroll-test",
            ),
        ])
        assert len(results) == 1
        assert results[0].outcome.success
        assert mouse.scroll.await_count == 3

    @pytest.mark.asyncio
    async def test_first_failure_aborts_remaining_steps(self) -> None:
        # First step succeeds (key_combo via keyboard), second fails
        # (no mouse for scroll). The executor should NOT call a third.
        kb = AsyncMock()
        ex = PlanExecutor(AgentContext(keyboard=kb))
        # Add a third sentinel step we expect to be skipped.
        sentinel_called = []

        results = await ex.run([
            PlanStep(agent="key_combo",
                     kwargs={"modifiers": ["super"], "key": "l"},
                     rationale="lock"),
            PlanStep(agent="scroll",
                     kwargs={"direction": "down", "amount": 2},
                     rationale="scroll"),
            PlanStep(agent="key_combo",
                     kwargs={"modifiers": ["ctrl"], "key": "x"},
                     rationale="should-not-fire"),
        ])
        # Only the first two ran; the third was skipped.
        assert len(results) == 2
        assert results[0].outcome.success
        assert not results[1].outcome.success
        assert sentinel_called == []

    @pytest.mark.asyncio
    async def test_open_app_step_uses_platform_adapter(self) -> None:
        from unittest.mock import MagicMock

        from handsneyes.platforms.base import AppHint

        kb = AsyncMock()
        platform = MagicMock()
        platform.canonicalise_app = MagicMock(
            return_value=AppHint(
                canonical="terminal", expect_substrings=("terminal",),
            )
        )
        platform.open_app = AsyncMock()
        ex = PlanExecutor(AgentContext(keyboard=kb, platform=platform))
        results = await ex.run([
            PlanStep(
                agent="open_app",
                kwargs={"app": "a terminal"},
                rationale="open-keyword match",
            ),
        ])
        assert len(results) == 1
        assert results[0].outcome.success
        platform.canonicalise_app.assert_called_once_with("a terminal")
        platform.open_app.assert_awaited_once()
        assert results[0].outcome.data["canonical"] == "terminal"

    @pytest.mark.asyncio
    async def test_open_app_step_needs_keyboard(self) -> None:
        from unittest.mock import MagicMock

        platform = MagicMock()
        ex = PlanExecutor(AgentContext(platform=platform))
        results = await ex.run([
            PlanStep(agent="open_app", kwargs={"app": "terminal"}, rationale=""),
        ])
        assert not results[0].outcome.success
        assert "no keyboard" in results[0].outcome.reason

    @pytest.mark.asyncio
    async def test_continue_on_failure_flag(self) -> None:
        kb = AsyncMock()
        ex = PlanExecutor(AgentContext(keyboard=kb))
        results = await ex.run(
            [
                PlanStep(agent="scroll",
                         kwargs={"direction": "down", "amount": 1},
                         rationale="will-fail"),
                PlanStep(agent="key_combo",
                         kwargs={"modifiers": ["super"], "key": "l"},
                         rationale="lock"),
            ],
            stop_on_failure=False,
        )
        assert len(results) == 2
        assert not results[0].outcome.success
        assert results[1].outcome.success
