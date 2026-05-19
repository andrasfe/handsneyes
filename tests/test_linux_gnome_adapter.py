"""Tests for the LinuxGnomeAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from handsneyes.platforms import load_adapter
from handsneyes.platforms.linux_gnome import (
    APP_ALIASES,
    LinuxGnomeAdapter,
)
from handsneyes.platforms.linux_gnome.aliases import canonicalise


def test_linux_gnome_registered() -> None:
    adapter = load_adapter("linux_gnome")
    assert isinstance(adapter, LinuxGnomeAdapter)
    assert adapter.name == "linux_gnome"


def test_capabilities_advertise_ubuntu_models() -> None:
    a = LinuxGnomeAdapter()
    caps = a.capabilities()
    assert caps.supports_activities_sweep is True
    assert caps.supports_launcher_typeahead is True
    assert "maximize" in caps.supports_window_intents
    assert caps.has_pointer_accel_model is True
    assert caps.has_longjump_model is True


def test_pointer_accel_checkpoint_resolves() -> None:
    a = LinuxGnomeAdapter()
    path = a.pointer_accel_checkpoint()
    assert path is not None
    assert (path / "weights.npz").exists()
    assert (path / "config.json").exists()


def test_longjump_checkpoint_resolves() -> None:
    a = LinuxGnomeAdapter()
    path = a.longjump_checkpoint()
    assert path is not None
    assert (path / "weights.npz").exists()


class TestAliases:
    def test_known_alias_maps(self) -> None:
        assert canonicalise("shell").canonical == "terminal"
        assert canonicalise("THE TERMINAL").canonical == "terminal"
        assert canonicalise("browser").canonical == "firefox"
        assert canonicalise("calc").canonical == "calculator"

    def test_unknown_alias_passes_through(self) -> None:
        h = canonicalise("steam")
        assert h.canonical == "steam"
        assert "steam" in h.expect_substrings

    def test_article_stripped(self) -> None:
        h = canonicalise("the obsidian")
        assert h.canonical == "obsidian"

    def test_aliases_dict_exposed(self) -> None:
        assert "firefox" in APP_ALIASES
        assert APP_ALIASES["chrome"].canonical == "google-chrome"


class TestRemapCombo:
    def test_linux_is_identity(self) -> None:
        a = LinuxGnomeAdapter()
        assert a.remap_combo(["ctrl"], "a") == (["ctrl"], "a")
        assert a.remap_combo(["ctrl", "shift"], "z") == (["ctrl", "shift"], "z")
        assert a.remap_combo([], "Return") == ([], "Return")


class TestOpenApp:
    @pytest.mark.asyncio
    async def test_open_app_sends_super_then_type_then_enter(
        self,
    ) -> None:
        a = LinuxGnomeAdapter()
        kb = AsyncMock()
        await a.open_app(
            kb,
            app=canonicalise("terminal"),
            settle_ms=0,
        )
        # Verify the sequence: Escape → Super (modifier-only) →
        # text → Enter
        kb.send_keystroke.assert_any_await("Escape")
        # Super-only tap uses send_key_combo(["super"], "")
        kb.send_key_combo.assert_any_await(["super"], "")
        kb.send_text.assert_any_await("terminal")
        kb.send_keystroke.assert_any_await("Enter")


class TestWindowAction:
    @pytest.mark.asyncio
    async def test_maximize_sends_super_up(self) -> None:
        a = LinuxGnomeAdapter()
        kb = AsyncMock()
        await a.window_action(kb, "maximize")
        kb.send_key_combo.assert_awaited_once_with(["super"], "Up")

    @pytest.mark.asyncio
    async def test_close_window_sends_alt_f4(self) -> None:
        a = LinuxGnomeAdapter()
        kb = AsyncMock()
        await a.window_action(kb, "close_window")
        kb.send_key_combo.assert_awaited_once_with(["alt"], "F4")

    @pytest.mark.asyncio
    async def test_unsupported_intent_raises(self) -> None:
        a = LinuxGnomeAdapter()
        kb = AsyncMock()
        with pytest.raises(NotImplementedError):
            await a.window_action(kb, "restore")


class TestFocusBrowser:
    @pytest.mark.asyncio
    async def test_first_attempt_activates_firefox(self) -> None:
        a = LinuxGnomeAdapter()
        from handsneyes.core.agents.context import AgentContext

        kb = AsyncMock()
        ctx = AgentContext(keyboard=kb)
        label = await a.focus_browser(ctx, attempt=1, max_attempts=4)
        assert label == "activated via firefox"
        kb.send_text.assert_any_await("firefox")

    @pytest.mark.asyncio
    async def test_no_keyboard_returns_clean_label(self) -> None:
        a = LinuxGnomeAdapter()
        from handsneyes.core.agents.context import AgentContext

        label = await a.focus_browser(
            AgentContext(), attempt=1, max_attempts=4,
        )
        assert "no-keyboard" in label


class TestHints:
    def test_login_hint_present(self) -> None:
        a = LinuxGnomeAdapter()
        hint = a.login_hint()
        assert hint is not None
        assert "GDM" in hint.description or "lock" in hint.description.lower()

    def test_cursor_theme_advice_present(self) -> None:
        a = LinuxGnomeAdapter()
        advice = a.cursor_theme_advice()
        assert advice is not None
        assert any("gsettings" in c for c in advice.shell_commands)
