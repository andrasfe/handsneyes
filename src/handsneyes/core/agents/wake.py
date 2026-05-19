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

import numpy as np

from handsneyes.core.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


# Mean-brightness threshold (0..1) below which the screen is treated
# as asleep / off / locked. A black screen is ~0; a typical desktop
# even with a dark-mode terminal is well above 0.10.
_ASLEEP_BRIGHTNESS_THRESHOLD = 0.06


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
            try:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(frame.image, label="wake_awake_check")
                brightness = float(np.asarray(frame.image).mean()) / 255.0
            except Exception as e:
                logger.debug("wake awake-check failed: %s", e)
                brightness = 0.0
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
