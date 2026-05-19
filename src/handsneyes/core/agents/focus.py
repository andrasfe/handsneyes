"""FocusAgent — bring the foreground app to a centred / maximised state.

Uses :class:`VerifyAgent` to decide visually whether the main
application window is "in focus and centred", and if not, takes a
sequence of corrective actions via the active
:class:`PlatformAdapter`:

  1. Send the WM "maximise focused window" intent.
  2. Re-verify after a brief settle.
  3. If still not centred, click in the image centre to give the
     window keyboard focus, then retry the maximise.
  4. Last resort, destructive: close-window then maximise whatever's
     now in front.

Each attempt re-verifies. We never click without first having a
visual confirmation, and we abort cleanly after ``max_attempts``.

The adapter chooses the actual chord — ``window_action("maximize")``
on linux_gnome is Super+Up; on macOS (which has no equivalent) the
agent falls back to ``window_action("fullscreen")`` (Ctrl+Cmd+F).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from handsneyes.core.agents.base import Agent, Outcome
from handsneyes.core.agents.verify import VerifyAgent

if TYPE_CHECKING:
    from handsneyes.platforms.base import WindowIntent

logger = logging.getLogger(__name__)


@dataclass
class FocusOutcome(Outcome):
    pass


_FOCUS_QUESTION = (
    "Look at the screen. Decide whether the foreground application "
    "window is maximised/centred and READY for interaction.\n\n"
    "IMPORTANT — these are NOT 'desktop borders' and you must NOT "
    "treat them as evidence the window isn't maximised:\n"
    "  * the system dock / taskbar / launcher (e.g. the strip of "
    "app icons on the left side of an Ubuntu/GNOME desktop)\n"
    "  * the top bar / menu bar / status bar at the very top\n"
    "  * the system tray / notification area at the bottom-right\n"
    "  * a vertical app-switcher strip\n"
    "These persistent OS chrome elements are present even on a "
    "fully maximised window. Their presence is normal.\n\n"
    "Answer FALSE only if ANY of these are true:\n"
    "  * the screen is black, dark, dim, blurred, or appears off/asleep\n"
    "  * the screen shows a screensaver, lock screen, or login prompt\n"
    "  * no clear application UI (text, controls, menus, content) is "
    "visible at all\n"
    "  * the foreground window is genuinely small (e.g. a floating "
    "popup that occupies less than half the area excluding the "
    "OS dock/taskbar) or sits in just one quadrant\n\n"
    "Answer TRUE when ALL of these are true:\n"
    "  * clear application UI is visible (text/controls/content)\n"
    "  * the foreground window fills most of the area NOT occupied "
    "by the OS dock/taskbar/menu bar\n"
    "  * the user could comfortably interact with it as the primary "
    "window right now"
)


_AWAKE_QUESTION = (
    "Is the screen currently showing clear, readable application "
    "content — i.e. NOT dark, blurred, off, asleep, on a screensaver, "
    "or on a lock/login screen? Answer true only if normal app UI "
    "is visible."
)


class FocusAgent(Agent):
    """Verify-then-fix the foreground window's centring."""

    name = "focus"

    async def run(  # type: ignore[override]
        self,
        *,
        max_attempts: int = 3,
        settle_seconds: float = 0.7,
        wake_first: bool = True,
    ) -> FocusOutcome:
        if self.ctx.keyboard is None:
            return FocusOutcome(
                success=False, reason="no keyboard in context",
            )

        verifier = VerifyAgent(self.ctx)

        # Awake check FIRST. Skip the wake stimuli (which include a
        # Down-arrow keystroke) when the screen is clearly bright —
        # avoids history-recall poisoning of foregrounded terminals.
        if wake_first:
            bright = await self._mean_brightness()
            if bright < 0.06:
                logger.info(
                    "FocusAgent: screen looks asleep (%.3f); waking",
                    bright,
                )
                await self._wake()
            else:
                logger.debug(
                    "FocusAgent: screen already bright (%.3f); skipping wake",
                    bright,
                )

        awake = await verifier.run(
            question=_AWAKE_QUESTION,
            visual_only=True,
            record_label="focus_awake_check",
        )
        if not awake:
            await self._wake()
            awake = await verifier.run(
                question=_AWAKE_QUESTION,
                visual_only=True,
                record_label="focus_awake_recheck",
            )
        if not awake:
            return FocusOutcome(
                success=False,
                reason=(
                    f"screen is not awake / showing no content "
                    f"({awake.reason}); won't act"
                ),
                data={"attempts": 0, "awake": False},
            )

        v0 = await verifier.run(
            question=_FOCUS_QUESTION,
            visual_only=True,
            record_label="focus_initial_check",
        )
        if v0:
            return FocusOutcome(
                success=True,
                reason=f"already focused: {v0.reason}",
                data={"attempts": 0},
            )

        for attempt in range(1, max_attempts + 1):
            await self._apply_action(attempt)
            await asyncio.sleep(settle_seconds)
            v = await verifier.run(
                question=_FOCUS_QUESTION,
                visual_only=True,
                record_label=f"focus_recheck_{attempt:02d}",
            )
            if v:
                return FocusOutcome(
                    success=True,
                    reason=f"focused after attempt {attempt}: {v.reason}",
                    data={"attempts": attempt},
                )

        return FocusOutcome(
            success=False,
            reason=f"still not centred/focused after {max_attempts} attempts",
            data={"attempts": max_attempts},
        )

    async def _mean_brightness(self) -> float:
        if self.ctx.capture is None:
            return 0.0
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception as e:  # noqa: BLE001
            logger.debug("brightness capture failed: %s", e)
            return 0.0
        return float(np.asarray(frame.image).mean()) / 255.0

    async def _wake(self) -> None:
        kb = self.ctx.keyboard
        mouse = self.ctx.mouse
        try:
            if mouse is not None:
                for _ in range(3):
                    await mouse.move(20, 0)
                    await asyncio.sleep(0.04)
                    await mouse.move(-20, 0)
                    await asyncio.sleep(0.04)
            if kb is not None:
                await kb.send_keystroke("Down")
        except Exception as e:  # noqa: BLE001
            logger.warning("Wake step failed: %s", e)
        await asyncio.sleep(0.8)

    async def _maximize_via_adapter(self) -> bool:
        """Call ``window_action("maximize")`` through the platform
        adapter; on macOS (which doesn't support a clean maximize)
        fall back to fullscreen. Returns True if the chord was sent.
        """
        assert self.ctx.keyboard is not None
        if self.ctx.platform is not None:
            caps = self.ctx.platform.capabilities()
            for intent in ("maximize", "fullscreen"):
                intent_t: WindowIntent = intent  # type: ignore[assignment,unused-ignore]
                if intent_t in caps.supports_window_intents:
                    try:
                        await self.ctx.platform.window_action(
                            self.ctx.keyboard, intent_t,
                        )
                        return True
                    except NotImplementedError:
                        continue
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "window_action(%r) failed: %s", intent, e,
                        )
                        return False
            return False
        # No adapter — fall back to the historical Linux/GNOME chord.
        try:
            await self.ctx.keyboard.send_key_combo(["super"], "Up")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("legacy Super+Up failed: %s", e)
            return False

    async def _apply_action(self, attempt: int) -> None:
        kb = self.ctx.keyboard
        mouse = self.ctx.mouse
        assert kb is not None
        if attempt == 1:
            try:
                await kb.send_keystroke("Escape")
                await asyncio.sleep(0.15)
            except Exception as e:  # noqa: BLE001
                logger.debug("Escape pre-maximise failed: %s", e)
            await self._maximize_via_adapter()
        elif attempt == 2:
            try:
                if mouse is not None:
                    for _ in range(6):
                        await mouse.move(80, 80)
                        await asyncio.sleep(0.02)
                    await mouse.click("left")
                    await asyncio.sleep(0.2)
            except Exception as e:  # noqa: BLE001
                logger.warning("Centre-click step failed: %s", e)
            await self._maximize_via_adapter()
        elif attempt == 3:
            logger.warning(
                "FocusAgent: attempt 3 — closing focused window "
                "(destructive last resort)"
            )
            if self.ctx.platform is not None and (
                "close_window"
                in self.ctx.platform.capabilities().supports_window_intents
            ):
                try:
                    await self.ctx.platform.window_action(
                        kb, "close_window",
                    )
                    await asyncio.sleep(0.5)
                except Exception as e:  # noqa: BLE001
                    logger.warning("close_window failed: %s", e)
            await self._maximize_via_adapter()
        else:
            await self._maximize_via_adapter()
