"""Browser-focus recovery via GNOME Activities sweep.

Ported pattern from terminaleyes/agents/navigate.py: tries firefox →
chrome → chromium by name through the Activities overview, falls back
to Super+1..9 number sweep (taskbar position). Each helper sends HID
through the supplied :class:`KeyboardOutput`; verification is the
caller's job.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from handsneyes.io.keyboard.base import KeyboardOutput

logger = logging.getLogger(__name__)


_BROWSER_NAMES: tuple[str, ...] = ("firefox", "chrome", "chromium")


async def activate_via_activities(kb: KeyboardOutput, name: str) -> None:
    """Super → type name → Enter. GNOME Activities overview path."""
    try:
        await kb.send_keystroke("Escape")
    except Exception as e:  # noqa: BLE001
        logger.debug("Escape before Activities failed: %s", e)
    await asyncio.sleep(0.15)
    # Bare Super tap → Activities. Pi backend treats empty key as
    # "modifier-only tap".
    await kb.send_key_combo(["super"], "")
    await asyncio.sleep(0.55)
    await kb.send_text(name)
    await asyncio.sleep(0.40)
    await kb.send_keystroke("Enter")


async def super_number_sweep(kb: KeyboardOutput) -> None:
    """Cycle through Super+1..Super+9 to surface a browser pinned to
    the GNOME dash. Last-ditch fallback when typed-name activation
    misses (e.g. wrong locale or already-running browser).
    """
    for n in range(1, 10):
        try:
            await kb.send_key_combo(["super"], str(n))
        except Exception as e:  # noqa: BLE001
            logger.debug("Super+%d failed: %s", n, e)
            return
        await asyncio.sleep(0.08)


async def try_activate_browser(
    kb: KeyboardOutput,
    *,
    candidates: tuple[str, ...] = _BROWSER_NAMES,
    attempt: int = 1,
) -> str:
    """One shot at bringing a browser foreground. Returns a label
    describing what was tried, for the caller's verify-loop logs.
    """
    if attempt <= len(candidates):
        name = candidates[attempt - 1]
        await activate_via_activities(kb, name)
        return f"activated via {name}"
    # Out of named candidates → fall back to dash-number sweep.
    await super_number_sweep(kb)
    return "Super+N sweep"
