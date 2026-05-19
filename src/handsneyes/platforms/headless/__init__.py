"""Headless platform adapter.

Used by tests, dry-runs, and any environment where no real OS-specific
behaviour can or should happen. Every method is a deterministic no-op
that surfaces enough signal for callers to verify they invoked it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from handsneyes.platforms.base import (
    AppHint,
    Capabilities,
    PlatformAdapter,
    WindowIntent,
)

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext
    from handsneyes.io.keyboard.base import KeyboardOutput

logger = logging.getLogger(__name__)


class HeadlessAdapter(PlatformAdapter):
    """Identity adapter. No HID, no OS calls, no models.

    - ``remap_combo`` returns input unchanged (identity).
    - ``canonicalise_app`` echoes the alias as the canonical name.
    - ``open_app`` / ``focus_browser`` / ``window_action`` no-op (the
      latter still respects the Capabilities check, raising on intents
      not in the supported set — which is empty by default).
    - Optional methods (cursor theme, model checkpoints, login hint)
      stay at ``None``.
    """

    name = "headless"
    display_name = "Headless (no-op)"
    package_root = Path(__file__).resolve().parent

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
        logger.debug(
            "headless.open_app no-op (app=%r, settle_ms=%d)",
            app.canonical,
            settle_ms,
        )

    async def focus_browser(
        self,
        ctx: AgentContext,
        *,
        attempt: int,
        max_attempts: int,
    ) -> str:
        logger.debug(
            "headless.focus_browser no-op (attempt=%d/%d)",
            attempt,
            max_attempts,
        )
        return "headless:noop"

    async def window_action(
        self,
        kb: KeyboardOutput,
        intent: WindowIntent,
    ) -> None:
        raise NotImplementedError(
            f"HeadlessAdapter does not implement window_action({intent!r}) "
            "— check capabilities().supports_window_intents first."
        )

    def remap_combo(
        self,
        modifiers: list[str],
        key: str,
    ) -> tuple[list[str], str]:
        return (list(modifiers), key)
