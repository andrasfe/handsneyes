"""Tests for tier-1 agents: VerifyAgent, CursorAgent, TargetAgent.

The tests stub out the vision-client / capture so they exercise the
agent control flow (preconditions, branch selection, outcome shapes)
without actually calling tesseract or a multimodal LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from handsneyes.core.agents.context import AgentContext
from handsneyes.core.agents.cursor import CursorAgent, CursorOutcome
from handsneyes.core.agents.target import TargetAgent, TargetOutcome
from handsneyes.core.agents.verify import VerifyAgent, VerifyOutcome


def _blank(w: int = 64, h: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 200, dtype=np.uint8)


# ─── VerifyAgent ────────────────────────────────────────────────────


class TestVerifyAgent:
    @pytest.mark.asyncio
    async def test_no_vision_client_fails(self) -> None:
        ctx = AgentContext()
        out = await VerifyAgent(ctx).run(question="is this a login screen?")
        assert isinstance(out, VerifyOutcome)
        assert not out
        assert "vision client" in out.reason

    @pytest.mark.asyncio
    async def test_no_capture_no_image_fails(self) -> None:
        ctx = AgentContext(vision_client=MagicMock(), vision_model="x")
        out = await VerifyAgent(ctx).run(question="x")
        assert not out
        assert "capture" in out.reason

    @pytest.mark.asyncio
    async def test_verify_yes_verdict(self) -> None:
        # Mock the chat client to return a JSON object with answer=true.
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"answer": true, "reason": "looks like one"}'
                        )
                    )
                ]
            )
        )
        ctx = AgentContext(vision_client=client, vision_model="m")
        img = _blank(128, 64)
        out = await VerifyAgent(ctx).run(question="login?", image=img)
        assert out.success
        assert "looks like one" in out.reason
        assert out.data["parsed"]["answer"] is True

    @pytest.mark.asyncio
    async def test_verify_no_verdict(self) -> None:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"answer": false, "reason": "browser focused"}'
                        )
                    )
                ]
            )
        )
        ctx = AgentContext(vision_client=client, vision_model="m")
        out = await VerifyAgent(ctx).run(question="login?", image=_blank())
        assert not out.success
        assert "browser" in out.reason

    @pytest.mark.asyncio
    async def test_verify_unparseable_falls_back_to_false(self) -> None:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content="not even json here")
                    )
                ]
            )
        )
        ctx = AgentContext(vision_client=client, vision_model="m")
        out = await VerifyAgent(ctx).run(question="login?", image=_blank())
        assert not out.success

    def test_extract_json_handles_empty(self) -> None:
        assert VerifyAgent._extract_json("") is None
        assert VerifyAgent._extract_json("no braces") is None
        assert VerifyAgent._extract_json('prefix {"a": 1} suffix') == {"a": 1}


# ─── CursorAgent ────────────────────────────────────────────────────


class TestCursorAgent:
    @pytest.mark.asyncio
    async def test_no_capture_no_image_fails(self) -> None:
        ctx = AgentContext()
        out = await CursorAgent(ctx).run()
        assert isinstance(out, CursorOutcome)
        assert not out
        assert "capture" in out.reason

    @pytest.mark.asyncio
    async def test_unknown_mode_fails(self) -> None:
        ctx = AgentContext()
        out = await CursorAgent(ctx).run(mode="invalid", image=_blank())
        # mode is checked after the capture guard; we passed image so
        # we hit the mode check at the end.
        assert not out
        assert "unknown mode" in out.reason

    @pytest.mark.asyncio
    async def test_hsv_mode_unverified_hits_blank(self) -> None:
        # Blank image → HSV finds nothing → returns failure on hsv mode.
        ctx = AgentContext()
        out = await CursorAgent(ctx).run(
            mode="hsv", image=_blank(200, 200), verify_motion=False,
        )
        assert not out
        assert "did not find" in out.reason


# ─── TargetAgent ────────────────────────────────────────────────────


class TestTargetAgent:
    @pytest.mark.asyncio
    async def test_empty_description_fails(self) -> None:
        ctx = AgentContext()
        out = await TargetAgent(ctx).run(description="")
        assert isinstance(out, TargetOutcome)
        assert not out
        assert "empty description" in out.reason

    @pytest.mark.asyncio
    async def test_no_capture_no_image_fails(self) -> None:
        ctx = AgentContext()
        out = await TargetAgent(ctx).run(description="the Run button")
        assert not out
        assert "no capture" in out.reason

    @pytest.mark.asyncio
    async def test_all_locators_miss_returns_failure(self) -> None:
        # No showui_query, no tesseract match expected on blank image.
        ctx = AgentContext()
        out = await TargetAgent(ctx).run(
            description="the Run button", image=_blank(),
        )
        assert not out
        assert "missed" in out.reason

    @pytest.mark.asyncio
    async def test_showui_direct_hit_returns_position(self) -> None:
        # Stub showui_query to return a fixed (x, y) for any prompt
        # — TargetAgent should pick it up via the direct branch.
        async def fake_showui(_b64: str, _prompt: str) -> tuple[float, float]:
            return (0.42, 0.66)

        ctx = AgentContext(showui_query=fake_showui)
        out = await TargetAgent(ctx).run(
            description='click the "Run" button', image=_blank(),
        )
        assert out
        assert out.data["position"] == (0.42, 0.66)
        # Either direct or cropped variant may have grounded; both are
        # ShowUI-method outcomes.
        assert out.data["method"].startswith("showui") or \
               out.data["method"].startswith("cropped_showui")
