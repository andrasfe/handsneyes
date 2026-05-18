"""Tests for the PlatformAdapter ABC and supporting dataclasses."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from handsneyes.platforms.base import (
    AppHint,
    Capabilities,
    CursorThemeAdvice,
    LoginHint,
    PlatformAdapter,
    WindowIntent,
)

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext
    from handsneyes.io.keyboard.base import KeyboardOutput


class _FakeAdapter(PlatformAdapter):
    """Minimal concrete adapter used to exercise the contract."""

    name = "fake"
    display_name = "Fake"

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
    ) -> None:
        return None

    async def focus_browser(
        self,
        ctx: AgentContext,
        *,
        attempt: int,
        max_attempts: int,
    ) -> str:
        return "fake:noop"

    async def window_action(
        self,
        kb: KeyboardOutput,
        intent: WindowIntent,
    ) -> None:
        raise NotImplementedError

    def remap_combo(
        self,
        modifiers: list[str],
        key: str,
    ) -> tuple[list[str], str]:
        return (list(modifiers), key)


def test_cannot_instantiate_abc_directly() -> None:
    with pytest.raises(TypeError):
        PlatformAdapter()  # type: ignore[abstract]


def test_concrete_subclass_with_minimal_overrides_works() -> None:
    adapter = _FakeAdapter()
    assert adapter.remap_combo(["ctrl"], "a") == (["ctrl"], "a")
    assert adapter.capabilities() == Capabilities()


def test_dataclasses_are_frozen() -> None:
    hint = AppHint(canonical="terminal")
    assert hint.canonical == "terminal"
    with pytest.raises(dataclasses.FrozenInstanceError):
        hint.canonical = "other"  # type: ignore[misc]


def test_capabilities_defaults() -> None:
    caps = Capabilities()
    assert caps.supports_activities_sweep is False
    assert caps.supports_window_intents == frozenset()
    assert caps.supports_launcher_typeahead is False
    assert caps.supports_cursor_theme_setup is False
    assert caps.has_pointer_accel_model is False
    assert caps.has_longjump_model is False


def test_optional_methods_default_to_none() -> None:
    adapter = _FakeAdapter()
    assert adapter.cursor_theme_advice() is None
    assert adapter.pointer_accel_checkpoint() is None
    assert adapter.longjump_checkpoint() is None
    assert adapter.login_hint() is None


def test_login_hint_dataclass_is_frozen() -> None:
    hint = LoginHint(
        description="GDM",
        password_field_cue="centered input",
        after_unlock_cue="desktop wallpaper",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hint.description = "other"  # type: ignore[misc]


def test_cursor_theme_advice_defaults_to_empty_tuples() -> None:
    advice = CursorThemeAdvice(summary="set redglass")
    assert advice.shell_commands == ()
    assert advice.manual_steps == ()
    assert advice.verify_question == ""
