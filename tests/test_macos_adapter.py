"""Tests for the MacOSAdapter skeleton."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from handsneyes.platforms import load_adapter
from handsneyes.platforms.macos import MacOSAdapter


def test_macos_registered() -> None:
    a = load_adapter("macos")
    assert isinstance(a, MacOSAdapter)
    assert a.name == "macos"


def test_capabilities_skeleton() -> None:
    a = MacOSAdapter()
    caps = a.capabilities()
    assert caps.supports_activities_sweep is False
    assert caps.supports_launcher_typeahead is True
    # No models ship by default.
    assert caps.has_pointer_accel_model is False
    assert caps.has_longjump_model is False


class TestAliases:
    def test_terminal_alias(self) -> None:
        a = MacOSAdapter()
        h = a.canonicalise_app("the terminal")
        assert h.canonical == "Terminal"

    def test_iterm_alias(self) -> None:
        a = MacOSAdapter()
        assert a.canonicalise_app("iterm").canonical == "iTerm"


class TestRemapCombo:
    def test_ctrl_a_becomes_cmd_a(self) -> None:
        a = MacOSAdapter()
        assert a.remap_combo(["ctrl"], "a") == (["cmd"], "a")

    def test_ctrl_z_becomes_cmd_z(self) -> None:
        a = MacOSAdapter()
        assert a.remap_combo(["ctrl"], "z") == (["cmd"], "z")

    def test_ctrl_uppercase_letter_remapped(self) -> None:
        a = MacOSAdapter()
        # The remap is case-insensitive on the key but preserves it.
        assert a.remap_combo(["ctrl"], "A") == (["cmd"], "A")

    def test_unknown_ctrl_letter_passes_through(self) -> None:
        a = MacOSAdapter()
        # Ctrl+Backspace is not in the safe-letter set
        assert a.remap_combo(["ctrl"], "Backspace") == (
            ["ctrl"], "Backspace",
        )

    def test_ctrl_shift_letter_passes_through(self) -> None:
        # Anything beyond just-ctrl is preserved (Ctrl-Shift-T,
        # Ctrl-Alt-X, etc.) — those have different macOS idioms.
        a = MacOSAdapter()
        assert a.remap_combo(["ctrl", "shift"], "t") == (
            ["ctrl", "shift"], "t",
        )

    def test_no_modifiers_passes_through(self) -> None:
        a = MacOSAdapter()
        assert a.remap_combo([], "Enter") == ([], "Enter")


class TestOpenApp:
    @pytest.mark.asyncio
    async def test_open_app_uses_spotlight(self) -> None:
        a = MacOSAdapter()
        kb = AsyncMock()
        await a.open_app(
            kb,
            app=a.canonicalise_app("safari"),
            settle_ms=0,
        )
        kb.send_key_combo.assert_any_await(["cmd"], "space")
        kb.send_text.assert_any_await("Safari")
        kb.send_keystroke.assert_any_await("Enter")


class TestWindowAction:
    @pytest.mark.asyncio
    async def test_fullscreen_uses_ctrl_cmd_f(self) -> None:
        a = MacOSAdapter()
        kb = AsyncMock()
        await a.window_action(kb, "fullscreen")
        kb.send_key_combo.assert_awaited_once_with(["ctrl", "cmd"], "f")

    @pytest.mark.asyncio
    async def test_close_window_uses_cmd_w(self) -> None:
        a = MacOSAdapter()
        kb = AsyncMock()
        await a.window_action(kb, "close_window")
        kb.send_key_combo.assert_awaited_once_with(["cmd"], "w")

    @pytest.mark.asyncio
    async def test_unsupported_intent_raises(self) -> None:
        a = MacOSAdapter()
        kb = AsyncMock()
        with pytest.raises(NotImplementedError):
            await a.window_action(kb, "maximize")  # use fullscreen on mac
