"""Tests for tier-2 agents: WakeAgent, TypeAgent, ScrollAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from handsneyes.core.agents.context import AgentContext
from handsneyes.core.agents.scroll import ScrollAgent, ScrollOutcome
from handsneyes.core.agents.type_text import TypeAgent
from handsneyes.core.agents.wake import WakeAgent, WakeOutcome


def _ctx_with_keyboard() -> tuple[AgentContext, AsyncMock]:
    kb = AsyncMock()
    return AgentContext(keyboard=kb), kb


def _ctx_with_mouse() -> tuple[AgentContext, AsyncMock]:
    m = AsyncMock()
    return AgentContext(mouse=m), m


class TestWake:
    @pytest.mark.asyncio
    async def test_wake_requires_mouse_or_keyboard(self) -> None:
        ctx = AgentContext()
        out = await WakeAgent(ctx).run(check_awake=False)
        assert isinstance(out, WakeOutcome)
        assert not out
        assert "no mouse or keyboard" in out.reason

    @pytest.mark.asyncio
    async def test_wake_jiggles_and_keystrokes(self) -> None:
        kb = AsyncMock()
        m = AsyncMock()
        ctx = AgentContext(keyboard=kb, mouse=m)
        out = await WakeAgent(ctx).run(
            jiggle_count=2,
            check_awake=False,
            settle_seconds=0.0,
        )
        assert out
        assert m.move.await_count == 4  # 2 jiggles × 2 moves (forward+back)
        kb.send_keystroke.assert_awaited_once_with("Down")
        m.click.assert_awaited_once_with("left")


class TestType:
    @pytest.mark.asyncio
    async def test_type_requires_keyboard(self) -> None:
        ctx = AgentContext()
        out = await TypeAgent(ctx).run(text="hello")
        assert not out
        assert "no keyboard" in out.reason

    @pytest.mark.asyncio
    async def test_type_empty_no_submit_fails(self) -> None:
        ctx, _ = _ctx_with_keyboard()
        out = await TypeAgent(ctx).run(text="")
        assert not out
        assert "empty text" in out.reason

    @pytest.mark.asyncio
    async def test_type_sends_text(self) -> None:
        ctx, kb = _ctx_with_keyboard()
        out = await TypeAgent(ctx).run(text="hello", post_settle=0.0)
        assert out
        kb.send_text.assert_awaited_once_with(
            "hello", secret=False, warmup=True,
        )

    @pytest.mark.asyncio
    async def test_type_submit_presses_enter(self) -> None:
        ctx, kb = _ctx_with_keyboard()
        out = await TypeAgent(ctx).run(
            text="ls", submit=True, post_settle=0.0,
        )
        assert out
        kb.send_text.assert_awaited_once()
        kb.send_keystroke.assert_awaited_once_with("Enter")

    @pytest.mark.asyncio
    async def test_type_secret_redacts_reason(self) -> None:
        ctx, _ = _ctx_with_keyboard()
        out = await TypeAgent(ctx).run(
            text="hunter2", secret=True, post_settle=0.0,
        )
        assert out
        assert "hunter2" not in out.reason
        assert "redacted" in out.reason
        assert "length=7" in out.reason


class TestScroll:
    @pytest.mark.asyncio
    async def test_scroll_requires_mouse(self) -> None:
        ctx = AgentContext()
        out = await ScrollAgent(ctx).run()
        assert not out
        assert "no mouse" in out.reason

    @pytest.mark.asyncio
    async def test_scroll_rejects_unknown_direction(self) -> None:
        ctx, _ = _ctx_with_mouse()
        out = await ScrollAgent(ctx).run(direction="sideways")
        assert not out
        assert "unknown direction" in out.reason

    @pytest.mark.asyncio
    async def test_scroll_rejects_non_positive_amount(self) -> None:
        ctx, _ = _ctx_with_mouse()
        out = await ScrollAgent(ctx).run(amount=0)
        assert not out

    @pytest.mark.asyncio
    async def test_scroll_down_calls_scroll(self) -> None:
        ctx, m = _ctx_with_mouse()
        out = await ScrollAgent(ctx).run(
            direction="down",
            amount=3,
            between_ticks=0.0,
            post_settle=0.0,
        )
        assert isinstance(out, ScrollOutcome)
        assert out
        assert m.scroll.await_count == 3
        assert m.scroll.call_args_list[0].args == (1,)
        assert out.data == {"direction": "down", "amount": 3}

    @pytest.mark.asyncio
    async def test_scroll_up_uses_negative_step(self) -> None:
        ctx, m = _ctx_with_mouse()
        await ScrollAgent(ctx).run(
            direction="up",
            amount=2,
            between_ticks=0.0,
            post_settle=0.0,
        )
        assert m.scroll.await_count == 2
        assert m.scroll.call_args_list[0].args == (-1,)
