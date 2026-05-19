"""Tests for tier-3 workflow agents: FocusAgent, NavigateAgent, ClickAgent.

Heavyweight integration (real homer, real LLM) is out of scope —
the homer's own behaviour is empirically validated against the live
target. These tests cover control flow + adapter wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from handsneyes.core.agents.click import ClickAgent, ClickOutcome, SearchAgent
from handsneyes.core.agents.context import AgentContext
from handsneyes.core.agents.focus import FocusAgent, FocusOutcome
from handsneyes.core.agents.navigate import NavigateAgent, NavigateOutcome
from handsneyes.platforms import load_adapter


def _verify_response(*, answer: bool, reason: str) -> MagicMock:
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=f'{{"answer": {str(answer).lower()}, '
                    f'"reason": "{reason}"}}'
                )
            )
        ]
    )


def _vision_client(*, answer: bool, reason: str = "ok") -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_verify_response(answer=answer, reason=reason),
    )
    return client


def _capture_blank() -> AsyncMock:
    import numpy as np
    cap = AsyncMock()
    cap.capture_frame = AsyncMock(
        return_value=MagicMock(
            image=np.full((128, 128, 3), 200, dtype=np.uint8),
        )
    )
    return cap


# ─── FocusAgent ─────────────────────────────────────────────────────


class TestFocusAgent:
    @pytest.mark.asyncio
    async def test_no_keyboard_fails(self) -> None:
        ctx = AgentContext()
        out = await FocusAgent(ctx).run(wake_first=False)
        assert isinstance(out, FocusOutcome)
        assert not out
        assert "no keyboard" in out.reason

    @pytest.mark.asyncio
    async def test_not_awake_short_circuits(self) -> None:
        client = _vision_client(answer=False, reason="dark screen")
        ctx = AgentContext(
            keyboard=AsyncMock(),
            mouse=AsyncMock(),
            capture=_capture_blank(),
            vision_client=client,
            vision_model="m",
        )
        out = await FocusAgent(ctx).run(
            wake_first=False, max_attempts=1, settle_seconds=0.0,
        )
        assert not out
        assert "not awake" in out.reason

    @pytest.mark.asyncio
    async def test_already_focused_returns_success(self) -> None:
        # Vision client says yes to BOTH the awake question and the
        # focus question.
        client = _vision_client(answer=True, reason="centred window")
        ctx = AgentContext(
            keyboard=AsyncMock(),
            mouse=AsyncMock(),
            capture=_capture_blank(),
            vision_client=client,
            vision_model="m",
        )
        out = await FocusAgent(ctx).run(
            wake_first=False, max_attempts=1, settle_seconds=0.0,
        )
        assert out
        assert "already focused" in out.reason
        assert out.data["attempts"] == 0

    @pytest.mark.asyncio
    async def test_maximize_via_adapter_uses_window_action(self) -> None:
        """When focus check fails, the agent calls
        platform.window_action('maximize')."""
        adapter = load_adapter("linux_gnome")
        kb = AsyncMock()
        ctx = AgentContext(keyboard=kb, platform=adapter)
        # Exercise the helper directly.
        ok = await FocusAgent(ctx)._maximize_via_adapter()
        assert ok
        kb.send_key_combo.assert_awaited_with(["super"], "Up")


# ─── NavigateAgent ──────────────────────────────────────────────────


class TestNavigateAgent:
    @pytest.mark.asyncio
    async def test_no_keyboard_fails(self) -> None:
        ctx = AgentContext()
        out = await NavigateAgent(ctx).run(url="https://example.com")
        assert isinstance(out, NavigateOutcome)
        assert not out
        assert "no keyboard" in out.reason

    @pytest.mark.asyncio
    async def test_empty_url_fails(self) -> None:
        ctx = AgentContext(keyboard=AsyncMock())
        out = await NavigateAgent(ctx).run(url="")
        assert not out
        assert "empty url" in out.reason

    @pytest.mark.asyncio
    async def test_no_adapter_when_recover_needed(self) -> None:
        """Browser not focused initially + no platform adapter →
        clean failure."""
        client = _vision_client(answer=False, reason="terminal foreground")
        ctx = AgentContext(
            keyboard=AsyncMock(),
            capture=_capture_blank(),
            vision_client=client,
            vision_model="m",
            # No platform set.
        )
        out = await NavigateAgent(ctx).run(
            url="https://example.com",
            max_focus_attempts=1,
            verify_after=False,
        )
        assert not out
        assert "no platform adapter" in out.reason

    @pytest.mark.asyncio
    async def test_already_browser_focused_types_url(self) -> None:
        """Pre-flight verify says yes → type the URL → skip post-flight."""
        client = _vision_client(answer=True, reason="browser foreground")
        kb = AsyncMock()
        ctx = AgentContext(
            keyboard=kb,
            capture=_capture_blank(),
            vision_client=client,
            vision_model="m",
            platform=load_adapter("linux_gnome"),
        )
        out = await NavigateAgent(ctx).run(
            url="https://example.com",
            verify_after=False,
            post_settle=0.0,
        )
        assert out
        # Ctrl+L sent to focus URL bar
        kb.send_key_combo.assert_any_await(["ctrl"], "l")
        # Enter pressed at the end
        kb.send_keystroke.assert_any_await("Enter")


# ─── ClickAgent ─────────────────────────────────────────────────────


class TestClickAgent:
    @pytest.mark.asyncio
    async def test_no_capture_fails(self) -> None:
        ctx = AgentContext(mouse=AsyncMock())
        out = await ClickAgent(ctx).run(target="Run button")
        assert isinstance(out, ClickOutcome)
        assert not out
        assert "no capture" in out.reason

    @pytest.mark.asyncio
    async def test_no_mouse_fails(self) -> None:
        ctx = AgentContext(capture=_capture_blank())
        out = await ClickAgent(ctx).run(target="Run button")
        assert not out
        assert "no mouse" in out.reason

    @pytest.mark.asyncio
    async def test_homer_clicked_returns_success(self) -> None:
        # Patch the homer to return a clicked outcome on the first try.
        ctx = AgentContext(
            capture=_capture_blank(),
            mouse=AsyncMock(),
            keyboard=AsyncMock(),
        )
        fake_homer_outcome = MagicMock(
            clicked=True,
            reason="clicked-ok",
            steps=3,
            proof_path=None,
        )
        fake_homer = MagicMock()
        fake_homer.run = AsyncMock(return_value=fake_homer_outcome)
        with patch(
            "handsneyes.core.vision.visual_servo_homer.VisualServoHomer",
            return_value=fake_homer,
        ):
            out = await ClickAgent(ctx).run(
                target="Run button", scroll_attempts=0,
            )
        assert out
        assert out.reason == "clicked-ok"
        assert out.data["scroll_attempts_used"] == 0

    @pytest.mark.asyncio
    async def test_homer_target_lost_triggers_scroll_retry(self) -> None:
        ctx = AgentContext(
            capture=_capture_blank(),
            mouse=AsyncMock(),
            keyboard=AsyncMock(),
        )
        # First call: target_lost. Second: target_lost. Third: clicked.
        outcomes = [
            MagicMock(clicked=False, reason="target_lost: not on screen",
                      steps=1, proof_path=None),
            MagicMock(clicked=False, reason="target_lost: still not on screen",
                      steps=1, proof_path=None),
            MagicMock(clicked=True, reason="clicked on second scroll",
                      steps=2, proof_path=None),
        ]
        fake_homer = MagicMock()
        fake_homer.run = AsyncMock(side_effect=outcomes)
        with patch(
            "handsneyes.core.vision.visual_servo_homer.VisualServoHomer",
            return_value=fake_homer,
        ):
            out = await ClickAgent(ctx).run(
                target="Run", scroll_attempts=2, scroll_amount=1,
                scroll_hover_at=None,
            )
        assert out
        assert out.data["scroll_attempts_used"] == 2

    def test_searchagent_alias(self) -> None:
        assert SearchAgent is ClickAgent
