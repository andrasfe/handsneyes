"""MacOSAdapter — skeleton macOS target adapter.

Implements the absolute minimum for `handsneyes do --target` to work
against a macOS target via Spotlight (Cmd+Space launcher) and the
Ctrl→Cmd shortcut remap. Browser-focus recovery is Cmd+Tab + dock
click. Window intents map to macOS green-button / Mission Control
shortcuts where they have clean equivalents.

No models ship by default — train per the docs/porting-to-new-os
runbook to drop weights into ``platforms.macos.models``.
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

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext
    from handsneyes.io.keyboard.base import KeyboardOutput

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _PACKAGE_ROOT / "models"

_SUPPORTED_WINDOW_INTENTS: frozenset[WindowIntent] = frozenset(
    {
        "fullscreen",     # Ctrl+Cmd+F
        "close_window",   # Cmd+W
        "next_window",    # Cmd+`
        "minimize",       # Cmd+M
        "show_desktop",   # F11 on most setups
    },
)


# Free-form alias map. Conservative — only obvious variants. The
# bundle_id field stays empty until someone wires AppleScript / `open
# -b` introspection (Phase B follow-up).
_APP_ALIASES: dict[str, AppHint] = {
    "terminal":     AppHint("Terminal", ("terminal",)),
    "the terminal": AppHint("Terminal", ("terminal",)),
    "iterm":        AppHint("iTerm", ("iterm",)),
    "iterm2":       AppHint("iTerm", ("iterm",)),
    "shell":        AppHint("Terminal", ("terminal",)),
    "finder":       AppHint("Finder", ("finder",)),
    "files":        AppHint("Finder", ("finder",)),
    "calculator":   AppHint("Calculator", ("calculator",)),
    "calc":         AppHint("Calculator", ("calculator",)),
    "safari":       AppHint("Safari", ("safari",)),
    "chrome":       AppHint("Google Chrome", ("chrome",)),
    "google chrome": AppHint("Google Chrome", ("chrome",)),
    "firefox":      AppHint("Firefox", ("firefox",)),
    "browser":      AppHint("Safari", ("safari", "chrome", "firefox")),
}


# Standard macOS shortcut chord remap: Ctrl-letter on the source
# semantic chord (the same chord a Linux user types) becomes Cmd-
# letter on macOS — preserving the *intent* (copy/cut/paste/select-
# all/save/find/etc.). The remap is intentionally narrow: only safe,
# unambiguous single-letter targets where Ctrl-letter is *almost
# always* the "Linux idiom" for what macOS spells Cmd-letter.
_CTRL_TO_CMD_LETTERS = frozenset(
    {
        "a", "c", "v", "x", "z", "y",
        "s", "f", "n", "o", "p", "w", "q",
        "b", "i", "u",  # bold/italic/underline
        "l",            # url bar
    },
)


class MacOSAdapter(PlatformAdapter):
    """macOS desktop adapter. Skeleton — fill in as features land."""

    name = "macos"
    display_name = "macOS"
    package_root = _PACKAGE_ROOT

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_activities_sweep=False,
            supports_window_intents=_SUPPORTED_WINDOW_INTENTS,
            supports_launcher_typeahead=True,
            supports_cursor_theme_setup=False,
            has_pointer_accel_model=(
                _MODELS_DIR / "pointer_accel" / "weights.npz"
            ).exists(),
            has_longjump_model=(
                _MODELS_DIR / "longjump" / "weights.npz"
            ).exists(),
        )

    def canonicalise_app(self, alias: str) -> AppHint:
        key = alias.strip().lower()
        if key in _APP_ALIASES:
            return _APP_ALIASES[key]
        # Unknown name: type it title-cased and expect a lowercased
        # match in the menu bar / window title.
        base = key
        return AppHint(canonical=base.title(), expect_substrings=(base,))

    async def open_app(
        self,
        kb: KeyboardOutput,
        *,
        app: AppHint,
        settle_ms: int = 1500,
    ) -> None:
        # Cmd+Space → Spotlight
        try:
            await kb.send_keystroke("Escape")
        except Exception as e:  # noqa: BLE001
            logger.debug("Escape before Spotlight failed: %s", e)
        await asyncio.sleep(0.15)
        await kb.send_key_combo(["cmd"], "space")
        await asyncio.sleep(0.55)
        await kb.send_text(app.canonical)
        await asyncio.sleep(0.45)
        await kb.send_keystroke("Enter")
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
            return "macos:no-keyboard"
        # First attempt: Cmd+Tab to cycle to the most recently used
        # app. If that wasn't a browser, subsequent attempts try
        # Spotlight by name.
        if attempt == 1:
            await ctx.keyboard.send_key_combo(["cmd"], "Tab")
            await asyncio.sleep(0.3)
            return "cmd-tab"
        candidates = ("Safari", "Google Chrome", "Firefox")
        if attempt - 2 < len(candidates):
            name = candidates[attempt - 2]
            await self.open_app(
                ctx.keyboard, app=AppHint(canonical=name), settle_ms=0,
            )
            return f"activated via spotlight:{name}"
        return "macos:exhausted"

    async def window_action(
        self,
        kb: KeyboardOutput,
        intent: WindowIntent,
    ) -> None:
        if intent not in _SUPPORTED_WINDOW_INTENTS:
            raise NotImplementedError(
                f"macos does not support window intent {intent!r}"
            )
        if intent == "fullscreen":
            await kb.send_key_combo(["ctrl", "cmd"], "f")
        elif intent == "close_window":
            await kb.send_key_combo(["cmd"], "w")
        elif intent == "next_window":
            await kb.send_key_combo(["cmd"], "`")
        elif intent == "minimize":
            await kb.send_key_combo(["cmd"], "m")
        elif intent == "show_desktop":
            await kb.send_keystroke("F11")

    def remap_combo(
        self,
        modifiers: list[str],
        key: str,
    ) -> tuple[list[str], str]:
        """Ctrl-letter → Cmd-letter on the standard shortcut set.

        Preserves combinations that aren't Linux idioms (Ctrl-Shift-T
        for new-tab, Ctrl-Alt-Backspace, etc.) by checking against an
        explicit allow-list of letters. Anything outside that
        list passes through unchanged.
        """
        # Only rewrite when ctrl is the SOLE modifier and the key is
        # in the safe-letter allow-list.
        if (
            "ctrl" in modifiers
            and len(modifiers) == 1
            and key.lower() in _CTRL_TO_CMD_LETTERS
        ):
            return (["cmd"], key)
        return (list(modifiers), key)

    def pointer_accel_checkpoint(self) -> Path | None:
        path = _MODELS_DIR / "pointer_accel"
        return path if (path / "weights.npz").exists() else None

    def longjump_checkpoint(self) -> Path | None:
        path = _MODELS_DIR / "longjump"
        return path if (path / "weights.npz").exists() else None

    def login_hint(self) -> LoginHint:
        return LoginHint(
            description=(
                "macOS lock screen — typically a translucent background, "
                "user avatar centred, then a single password field."
            ),
            password_field_cue=(
                "single centred password input, often with the user's "
                "avatar above it"
            ),
            after_unlock_cue=(
                "macOS desktop with the menu bar at the top "
                "(Apple logo left, app name, then menus)"
            ),
        )


__all__ = ["MacOSAdapter"]
