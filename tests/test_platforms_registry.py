"""Tests for the platforms entry-point registry."""

from __future__ import annotations

import pytest

from handsneyes.platforms import (
    UnknownPlatformError,
    available_platforms,
    load_adapter,
)
from handsneyes.platforms.base import PlatformAdapter
from handsneyes.platforms.headless import HeadlessAdapter


def test_headless_is_registered() -> None:
    assert "headless" in available_platforms()


def test_load_adapter_returns_concrete_instance() -> None:
    adapter = load_adapter("headless")
    assert isinstance(adapter, HeadlessAdapter)
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.name == "headless"


def test_load_unknown_raises() -> None:
    with pytest.raises(UnknownPlatformError, match="No platform adapter"):
        load_adapter("definitely_not_a_real_platform")


def test_env_override_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSNEYES_PLATFORM", "headless")
    adapter = load_adapter("anything_else_at_all")
    assert isinstance(adapter, HeadlessAdapter)


def test_env_override_for_unknown_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSNEYES_PLATFORM", "no_such_adapter_anywhere")
    with pytest.raises(UnknownPlatformError):
        load_adapter("headless")


class TestHeadlessAdapter:
    def test_remap_combo_is_identity(self) -> None:
        adapter = load_adapter("headless")
        assert adapter.remap_combo(["ctrl"], "a") == (["ctrl"], "a")
        assert adapter.remap_combo([], "Return") == ([], "Return")

    def test_canonicalise_app_echoes(self) -> None:
        adapter = load_adapter("headless")
        hint = adapter.canonicalise_app("terminal")
        assert hint.canonical == "terminal"

    @pytest.mark.asyncio
    async def test_open_app_noop(self) -> None:
        adapter = load_adapter("headless")
        await adapter.open_app(kb=None, app=adapter.canonicalise_app("x"))  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_focus_browser_returns_label(self) -> None:
        adapter = load_adapter("headless")
        label = await adapter.focus_browser(
            ctx=None,  # type: ignore[arg-type]
            attempt=1,
            max_attempts=3,
        )
        assert label == "headless:noop"

    @pytest.mark.asyncio
    async def test_window_action_raises_with_clear_message(self) -> None:
        adapter = load_adapter("headless")
        with pytest.raises(NotImplementedError, match="window_action"):
            await adapter.window_action(kb=None, intent="maximize")  # type: ignore[arg-type]
