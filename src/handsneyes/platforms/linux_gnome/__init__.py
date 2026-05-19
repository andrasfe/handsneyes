"""LinuxGnomeAdapter — the original supported platform.

Implements every PlatformAdapter method using the GNOME Activities
overview as launcher, GNOME window-management shortcuts (Super+Up to
maximize), and the redglass cursor theme for HSV-friendly detection.

Ships the Ubuntu-libinput-adaptive pointer_accel-v5 + longjump-v2
checkpoints under :mod:`platforms.linux_gnome.models`. The homer
loads them via :meth:`pointer_accel_checkpoint` /
:meth:`longjump_checkpoint`.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from handsneyes.platforms.base import (
    AppHint,
    Capabilities,
    LoginHint,
    PlatformAdapter,
    WindowIntent,
)
from handsneyes.platforms.linux_gnome.aliases import (
    APP_ALIASES,
    canonicalise,
)
from handsneyes.platforms.linux_gnome.browser_focus import (
    activate_via_activities,
    try_activate_browser,
)
from handsneyes.platforms.linux_gnome.cursor_theme import advice as _cursor_advice

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext
    from handsneyes.io.keyboard.base import KeyboardOutput
    from handsneyes.platforms.base import CursorThemeAdvice

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _PACKAGE_ROOT / "models"

_SUPPORTED_WINDOW_INTENTS: frozenset[WindowIntent] = frozenset(
    {
        "maximize",
        "minimize",
        "fullscreen",
        "close_window",
        "next_window",
        "prev_window",
        "show_desktop",
    },
)


class LinuxGnomeAdapter(PlatformAdapter):
    """Ubuntu/GNOME desktop adapter. Default platform for handsneyes."""

    name = "linux_gnome"
    display_name = "Linux / GNOME"
    package_root = _PACKAGE_ROOT

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_activities_sweep=True,
            supports_window_intents=_SUPPORTED_WINDOW_INTENTS,
            supports_launcher_typeahead=True,
            supports_cursor_theme_setup=True,
            has_pointer_accel_model=(
                _MODELS_DIR / "pointer_accel" / "weights.npz"
            ).exists(),
            has_longjump_model=(
                _MODELS_DIR / "longjump" / "weights.npz"
            ).exists(),
        )

    def canonicalise_app(self, alias: str) -> AppHint:
        return canonicalise(alias)

    async def open_app(
        self,
        kb: KeyboardOutput,
        *,
        app: AppHint,
        settle_ms: int = 1500,
    ) -> None:
        await activate_via_activities(kb, app.canonical)
        if settle_ms > 0:
            await asyncio.sleep(settle_ms / 1000.0)

    async def focus_browser(
        self,
        ctx: AgentContext,
        *,
        attempt: int,
        max_attempts: int,
    ) -> str:
        if ctx.keyboard is None:
            return "linux_gnome:no-keyboard"
        return await try_activate_browser(
            ctx.keyboard, attempt=attempt,
        )

    async def window_action(
        self,
        kb: KeyboardOutput,
        intent: WindowIntent,
    ) -> None:
        if intent not in _SUPPORTED_WINDOW_INTENTS:
            raise NotImplementedError(
                f"linux_gnome does not support window intent {intent!r}"
            )
        if intent == "maximize":
            await kb.send_key_combo(["super"], "Up")
        elif intent == "minimize":
            await kb.send_key_combo(["super"], "h")
        elif intent == "fullscreen":
            await kb.send_keystroke("F11")
        elif intent == "close_window":
            await kb.send_key_combo(["alt"], "F4")
        elif intent == "next_window":
            await kb.send_key_combo(["alt"], "Tab")
        elif intent == "prev_window":
            await kb.send_key_combo(["alt", "shift"], "Tab")
        elif intent == "show_desktop":
            await kb.send_key_combo(["super"], "d")

    def remap_combo(
        self,
        modifiers: list[str],
        key: str,
    ) -> tuple[list[str], str]:
        return (list(modifiers), key)

    def cursor_theme_advice(self) -> CursorThemeAdvice:
        return _cursor_advice()

    def pointer_accel_checkpoint(self) -> Path | None:
        path = _MODELS_DIR / "pointer_accel"
        return path if (path / "weights.npz").exists() else None

    def longjump_checkpoint(self) -> Path | None:
        path = _MODELS_DIR / "longjump"
        return path if (path / "weights.npz").exists() else None

    def login_hint(self) -> LoginHint:
        return LoginHint(
            description=(
                "GNOME Display Manager (GDM) lock screen — typically "
                "shows a clock, blurred background, user avatar in "
                "the centre, then a password input."
            ),
            password_field_cue=(
                "single centred password input, often with hidden "
                "dots/circles instead of plain text"
            ),
            after_unlock_cue=(
                "GNOME desktop wallpaper with the top status bar "
                "(Activities button left, clock centre, system "
                "menu right)"
            ),
        )


__all__ = ["APP_ALIASES", "LinuxGnomeAdapter"]
