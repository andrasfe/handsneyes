"""ClickAgent — find a target by description and click it.

The user-facing tier-3 click engine. Wraps
:class:`VisualServoHomer` with a scroll-and-retry loop: if the homer
can't locate the target on the first try, scroll the page (down by
default) and try again, up to ``scroll_attempts`` times.

``SearchAgent`` is preserved as a back-compat alias.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from handsneyes.core.agents.base import Agent, Outcome
from handsneyes.core.agents.scroll import ScrollAgent

logger = logging.getLogger(__name__)


@dataclass
class ClickOutcome(Outcome):
    pass


# Homer-reported reasons indicating "target wasn't on screen" — the
# only reasons a scroll-and-retry could help.
_TARGET_NOT_FOUND_REASONS = ("target_lost",)


class ClickAgent(Agent):
    """Find a target by description and click it via the visual servo."""

    name = "click"

    async def run(  # type: ignore[override]
        self,
        *,
        target: str,
        button: str = "left",
        scroll_attempts: int = 3,
        scroll_direction: str = "down",
        scroll_amount: int = 4,
        scroll_hover_at: tuple[float, float] | None = (0.5, 0.5),
    ) -> ClickOutcome:
        if self.ctx.capture is None:
            return ClickOutcome(
                success=False, reason="no capture in context",
            )
        if self.ctx.mouse is None:
            return ClickOutcome(
                success=False, reason="no mouse in context",
            )
        from handsneyes.core.vision.session_adapter import SessionAdapter
        from handsneyes.core.vision.visual_servo_homer import (
            VisualServoHomer,
        )

        adapter = SessionAdapter(self.ctx)
        last_outcome = None
        for attempt in range(scroll_attempts + 1):
            if attempt > 0:
                logger.info(
                    "ClickAgent: target not located; scroll "
                    "%d/%d (%s x%d)",
                    attempt,
                    scroll_attempts,
                    scroll_direction,
                    scroll_amount,
                )
                await ScrollAgent(self.ctx).run(
                    direction=scroll_direction,
                    amount=scroll_amount,
                    hover_at=scroll_hover_at,
                )
                await asyncio.sleep(0.4)

            # Fresh homer per attempt — internal state is stale after
            # a scroll (target image position has changed).
            homer = VisualServoHomer(session=adapter)
            outcome = await homer.run(target, button=button)
            last_outcome = outcome
            if outcome.clicked:
                return ClickOutcome(
                    success=True,
                    reason=outcome.reason,
                    data={
                        "steps": outcome.steps,
                        "proof_path": str(outcome.proof_path)
                        if outcome.proof_path else None,
                        "scroll_attempts_used": attempt,
                    },
                )
            # Only retry if the homer says the target wasn't located.
            reason_key = (outcome.reason or "").split(":", 1)[0]
            if reason_key not in _TARGET_NOT_FOUND_REASONS:
                break

        return ClickOutcome(
            success=False,
            reason=(
                last_outcome.reason if last_outcome is not None
                else "click failed"
            ),
            data={
                "scroll_attempts_used": scroll_attempts,
                "proof_path": (
                    str(last_outcome.proof_path)
                    if last_outcome and last_outcome.proof_path
                    else None
                ),
            },
        )


# Back-compat alias — preserve the SearchAgent name terminaleyes used.
SearchAgent = ClickAgent
