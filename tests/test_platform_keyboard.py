"""Tests for the PlatformKeyboard proxy."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from handsneyes.core.agents.context import AgentContext
from handsneyes.io.keyboard import PlatformKeyboard
from handsneyes.io.keyboard.base import KeyboardOutput
from handsneyes.platforms import load_adapter
from handsneyes.platforms.base import (
    AppHint,
    Capabilities,
    PlatformAdapter,
    WindowIntent,
)

if TYPE_CHECKING:
    from handsneyes.io.keyboard.base import KeyboardOutput as _Kb  # noqa: F401


class _SwapAdapter(PlatformAdapter):
    """Adapter that always replaces ctrl with cmd to exercise the proxy."""

    name = "swap"
    display_name = "swap"

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def canonicalise_app(self, alias: str) -> AppHint:
        return AppHint(canonical=alias)

    async def open_app(
        self,
        kb: KeyboardOutput,
        *,
        app: AppHint,
        settle_ms: int = 1500,
    ) -> None: ...

    async def focus_browser(
        self,
        ctx: AgentContext,
        *,
        attempt: int,
        max_attempts: int,
    ) -> str:
        return "swap:noop"

    async def window_action(
        self,
        kb: KeyboardOutput,
        intent: WindowIntent,
    ) -> None: ...

    def remap_combo(
        self,
        modifiers: list[str],
        key: str,
    ) -> tuple[list[str], str]:
        return ([("cmd" if m == "ctrl" else m) for m in modifiers], key)


@pytest.fixture
def inner() -> AsyncMock:
    fake = AsyncMock(spec=KeyboardOutput)
    return fake


class TestPassThrough:
    @pytest.mark.asyncio
    async def test_keystroke_passes_through(self, inner: AsyncMock) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_keystroke("Enter")
        inner.send_keystroke.assert_awaited_once_with("Enter")

    @pytest.mark.asyncio
    async def test_text_passes_through(self, inner: AsyncMock) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_text("hello")
        inner.send_text.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_text_forwards_kwargs(self, inner: AsyncMock) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_text("hunter2", secret=True, warmup=False)
        inner.send_text.assert_awaited_once_with(
            "hunter2", secret=True, warmup=False
        )

    @pytest.mark.asyncio
    async def test_send_line_uses_proxy(self, inner: AsyncMock) -> None:
        # send_line is inherited from KeyboardOutput; calling it should
        # go through the proxy's own send_text + send_keystroke.
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_line("ls -la")
        inner.send_text.assert_awaited_once_with("ls -la")
        inner.send_keystroke.assert_awaited_once_with("Enter")

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, inner: AsyncMock) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.connect()
        inner.connect.assert_awaited_once()
        await kb.disconnect()
        inner.disconnect.assert_awaited_once()


class TestRemapCombo:
    @pytest.mark.asyncio
    async def test_remap_applied_to_send_key_combo(
        self, inner: AsyncMock
    ) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_key_combo(["ctrl"], "a")
        inner.send_key_combo.assert_awaited_once_with(["cmd"], "a")

    @pytest.mark.asyncio
    async def test_remap_passes_unaffected_modifiers(
        self, inner: AsyncMock
    ) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_key_combo(["alt", "shift"], "F4")
        inner.send_key_combo.assert_awaited_once_with(
            ["alt", "shift"], "F4"
        )

    @pytest.mark.asyncio
    async def test_send_raw_combo_bypasses_adapter(
        self, inner: AsyncMock
    ) -> None:
        kb = PlatformKeyboard(inner, _SwapAdapter())
        await kb.send_raw_combo(["ctrl"], "a")
        inner.send_key_combo.assert_awaited_once_with(["ctrl"], "a")


class TestWithHeadless:
    @pytest.mark.asyncio
    async def test_headless_remap_is_identity(self, inner: AsyncMock) -> None:
        kb = PlatformKeyboard(inner, load_adapter("headless"))
        await kb.send_key_combo(["ctrl"], "c")
        inner.send_key_combo.assert_awaited_once_with(["ctrl"], "c")


def test_agent_context_has_platform_slot() -> None:
    ctx = AgentContext()
    assert ctx.platform is None
    ctx2 = AgentContext(platform=load_adapter("headless"))
    assert ctx2.platform is not None
    assert ctx2.platform.name == "headless"
