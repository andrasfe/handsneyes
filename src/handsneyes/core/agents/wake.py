"""WakeAgent — bring a sleeping screen / monitor to a usable state.

Sends a sequence of stimuli that wake monitors and dismiss screensaver
or clock overlays without triggering destructive shortcuts. Used by
LoginAgent and FocusAgent. Standalone-callable.

Skips entirely when the screen is already clearly awake. The Down-
arrow keystroke is a *targeted* GDM-overlay dismissal — when sent
to an awake desktop with a terminal/editor in focus, bash readline
(or vim) interprets it as a real key event, which corrupts whatever
text is being typed by subsequent steps. Self-check before acting
makes ``wake`` truly idempotent and safe to include in any plan.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from handsneyes.core.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


# Mean-brightness threshold (0..1) below which the screen is treated
# as asleep / off / locked. A black screen is ~0; a typical desktop
# even with a dark-mode terminal is well above 0.10.
_ASLEEP_BRIGHTNESS_THRESHOLD = 0.06


def _looks_like_test_pattern(img: np.ndarray) -> bool:
    """macOS' "camera unavailable" placeholder is a static SMPTE-style
    bar pattern with perfectly vertical bands. Real captured screens
    have icons, text, window edges → high vertical gradient. Catching
    this state prevents downstream agents from operating on garbage:
    a typical bar pattern has mean brightness ~0.5 which passes the
    asleep check but contains no real UI to verify against."""
    if img is None or img.size == 0:
        return False
    gray = (
        img if img.ndim == 2
        else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    )
    vert_grad = float(np.abs(np.diff(gray.astype(np.int16), axis=0)).mean())
    return vert_grad < 1.5


@dataclass
class WakeOutcome(Outcome):
    pass


class WakeAgent(Agent):
    """Mouse jiggle + arrow key + click. Idempotent; safe to retry.

    Self-checks for awake-ness via the captured frame's mean
    brightness when ``check_awake=True`` (default). Skips all
    actions on a clearly-awake screen so the Down-arrow doesn't
    bleed into a foregrounded terminal / editor.
    """

    name = "wake"

    async def run(  # type: ignore[override]
        self,
        *,
        jiggle_count: int = 4,
        send_arrow: bool = True,
        click: bool = True,
        settle_seconds: float = 0.6,
        check_awake: bool = True,
    ) -> WakeOutcome:
        if self.ctx.mouse is None and self.ctx.keyboard is None:
            return WakeOutcome(
                success=False, reason="no mouse or keyboard in context",
            )

        # 0. Awake self-check. Cheap — one frame + one numpy mean.
        if check_awake and self.ctx.capture is not None:
            captured_img = None
            try:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(frame.image, label="wake_awake_check")
                captured_img = frame.image
                brightness = float(np.asarray(captured_img).mean()) / 255.0
            except Exception as e:
                logger.debug("wake awake-check failed: %s", e)
                brightness = 0.0
            # SMPTE-bar detection: brightness alone passes when the
            # webcam is on the macOS "camera unavailable" placeholder
            # (mid-gray average ~0.5). Downstream agents that operate
            # on garbage frames silently "succeed" — LoginAgent reports
            # the screen unlocked, click_at misses every target, etc.
            # Fail fast with a real error instead.
            if captured_img is not None and _looks_like_test_pattern(
                captured_img,
            ):
                msg = (
                    "webcam is returning the test pattern — camera busy "
                    "or unavailable. Replug the USB cam, free it from "
                    "another app (Photo Booth / Slack / Zoom / Meet), "
                    "or check macOS Settings → Privacy & Security → "
                    "Camera permission. Aborting wake."
                )
                logger.error("WakeAgent: %s", msg)
                return WakeOutcome(success=False, reason=msg)
            if brightness >= _ASLEEP_BRIGHTNESS_THRESHOLD:
                msg = (
                    f"screen already awake (brightness={brightness:.3f}); "
                    "skipping wake stimuli"
                )
                logger.info("WakeAgent: %s", msg)
                return WakeOutcome(success=True, reason=msg)

        # 1. Mouse jiggle — wakes monitors and registers activity.
        if self.ctx.mouse is not None:
            for _ in range(jiggle_count):
                try:
                    await self.ctx.mouse.move(20, 0)
                    await asyncio.sleep(0.04)
                    await self.ctx.mouse.move(-20, 0)
                    await asyncio.sleep(0.04)
                except Exception as e:
                    logger.warning("Wake jiggle failed: %s", e)
                    break

        # 2. Down arrow — dismisses GDM clock overlay; safe key.
        if send_arrow and self.ctx.keyboard is not None:
            try:
                await self.ctx.keyboard.send_keystroke("Down")
            except Exception as e:
                logger.warning("Wake keystroke failed: %s", e)
        await asyncio.sleep(0.4)

        # 3. Left click — covers lock screens that need a click before
        # showing the password prompt.
        if click and self.ctx.mouse is not None:
            try:
                await self.ctx.mouse.click("left")
            except Exception as e:
                logger.warning("Wake click failed: %s", e)
        await asyncio.sleep(settle_seconds)

        return WakeOutcome(success=True, reason="wake sequence completed")
