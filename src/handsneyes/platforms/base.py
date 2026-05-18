"""PlatformAdapter — every OS-specific assumption flows through here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext
    from handsneyes.io.keyboard.base import KeyboardOutput


WindowIntent = Literal[
    "maximize",
    "minimize",
    "fullscreen",
    "restore",
    "close_window",
    "next_window",
    "prev_window",
    "show_desktop",
]


@dataclass(frozen=True)
class AppHint:
    """Canonical app id + OCR/visual cues that prove it's foreground."""

    canonical: str
    expect_substrings: tuple[str, ...] = ()
    visual_cue: str = ""
    bundle_id: str = ""


@dataclass(frozen=True)
class LoginHint:
    """Visual cues a VerifyAgent can ask about to recognise the lock screen."""

    description: str
    password_field_cue: str
    after_unlock_cue: str


@dataclass(frozen=True)
class CursorThemeAdvice:
    """Declarative recipe the operator runs ON THE TARGET (not via HID)."""

    summary: str
    shell_commands: tuple[str, ...] = ()
    manual_steps: tuple[str, ...] = ()
    verify_question: str = ""


@dataclass(frozen=True)
class Capabilities:
    """Adapter feature flags. Defaults are pessimistic."""

    supports_activities_sweep: bool = False
    supports_window_intents: frozenset[WindowIntent] = field(
        default_factory=frozenset
    )
    supports_launcher_typeahead: bool = False
    supports_cursor_theme_setup: bool = False
    has_pointer_accel_model: bool = False
    has_longjump_model: bool = False


class PlatformAdapter(ABC):
    """OS-specific behaviour for one target. One instance per active target."""

    name: str = ""
    display_name: str = ""
    package_root: Path = Path()

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Static feature flags consulted by agents before calling optional methods."""

    @abstractmethod
    def canonicalise_app(self, alias: str) -> AppHint:
        """Map a free-form name ('shell', 'the terminal') to a canonical AppHint."""

    @abstractmethod
    async def open_app(
        self,
        kb: KeyboardOutput,
        *,
        app: AppHint,
        settle_ms: int = 1500,
    ) -> None:
        """Open the launcher, type the app name, press Enter. Does NOT verify."""

    @abstractmethod
    async def focus_browser(
        self,
        ctx: AgentContext,
        *,
        attempt: int,
        max_attempts: int,
    ) -> str:
        """One corrective shot to bring a browser foreground. Returns a method label."""

    @abstractmethod
    async def window_action(
        self,
        kb: KeyboardOutput,
        intent: WindowIntent,
    ) -> None:
        """Execute the WM intent. Raise NotImplementedError if not in capabilities()."""

    @abstractmethod
    def remap_combo(
        self,
        modifiers: list[str],
        key: str,
    ) -> tuple[list[str], str]:
        """Pure function: rewrite a logical chord into the OS-native chord."""

    def cursor_theme_advice(self) -> CursorThemeAdvice | None:
        """Optional. Return None when the adapter has nothing to recommend."""
        return None

    def pointer_accel_checkpoint(self) -> Path | None:
        """Directory containing config.json + weights.npz, or None."""
        return None

    def longjump_checkpoint(self) -> Path | None:
        """Long-jump model directory, or None when no checkpoint ships."""
        return None

    def login_hint(self) -> LoginHint | None:
        """Hints for VerifyAgent when checking the lockscreen / wake state."""
        return None
