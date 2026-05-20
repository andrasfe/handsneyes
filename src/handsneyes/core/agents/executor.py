"""PlanExecutor — turn a list of PlanSteps into actual agent invocations.

Phase B addition that lets ``handsneyes do "<intent>"`` execute,
rather than just print, the planner's output. Each step's ``agent``
key maps to a concrete :class:`Agent` subclass; ``kwargs`` are passed
through directly.

Agents that need real I/O (capture, mouse, keyboard, vision client)
fail cleanly when their resources are absent — the executor is
purely a dispatcher; it doesn't build the AgentContext.

Phase B Block 3 scope: dispatch table + linear execution. Phase C's
Command Center hands a per-run AgentContext into the executor and
collects step outcomes for the UI's history pane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from handsneyes.core.agents.base import Outcome
from handsneyes.core.agents.click import ClickAgent
from handsneyes.core.agents.controller import (
    PlanStep,  # noqa: TC001 — runtime-typed dataclass param
)
from handsneyes.core.agents.focus import FocusAgent
from handsneyes.core.agents.login import LoginAgent
from handsneyes.core.agents.navigate import NavigateAgent
from handsneyes.core.agents.scroll import ScrollAgent
from handsneyes.core.agents.type_text import TypeAgent
from handsneyes.core.agents.wake import WakeAgent

if TYPE_CHECKING:
    from handsneyes.core.agents.base import Agent
    from handsneyes.core.agents.context import AgentContext

logger = logging.getLogger(__name__)


# ─── dispatch table ────────────────────────────────────────────────


def _make_key_combo_agent_step(ctx: AgentContext, mods: list[str], key: str) -> Outcome:
    """Synthetic step that sends a chord directly via the keyboard.

    The rule planner emits ``agent="key_combo"`` for ``lock`` and
    similar one-shot chord intents. We dispatch this as a tiny
    inline coroutine rather than introducing a separate agent class.
    """
    raise NotImplementedError("Should be called via _key_combo_runner async")


async def _key_combo_runner(
    ctx: AgentContext, *, modifiers: list[str], key: str,
) -> Outcome:
    if ctx.keyboard is None:
        return Outcome(success=False, reason="no keyboard in context")
    try:
        await ctx.keyboard.send_key_combo(modifiers, key)
    except Exception as e:  # noqa: BLE001
        return Outcome(success=False, reason=f"send_key_combo failed: {e}")
    return Outcome(
        success=True,
        reason=f"sent {'+'.join(modifiers)}+{key}",
        data={"modifiers": modifiers, "key": key},
    )


_AGENT_CLASSES: dict[str, type[Agent]] = {
    "wake":     WakeAgent,
    "type":     TypeAgent,
    "scroll":   ScrollAgent,
    "focus":    FocusAgent,
    "navigate": NavigateAgent,
    "login":    LoginAgent,
    "click":    ClickAgent,
}


@dataclass
class StepResult:
    """One executed plan step + its outcome."""

    step: PlanStep
    outcome: Outcome

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step.as_dict(),
            "outcome": {
                "success": bool(self.outcome.success),
                "reason": self.outcome.reason,
                "data": self.outcome.data,
            },
        }


class PlanExecutor:
    """Linear runner: each step in turn, abort on first failure."""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    async def run(
        self, plan: list[PlanStep], *, stop_on_failure: bool = True,
    ) -> list[StepResult]:
        results: list[StepResult] = []
        for step in plan:
            # Pre-step snapshot so the UI's frame pane has something to
            # show. Agents that capture their own frames (verify, focus,
            # navigate) drop additional frames in the same dir.
            await self._snapshot(f"executor_pre_{step.agent}")
            outcome = await self._run_step(step)
            await self._snapshot(f"executor_post_{step.agent}")
            results.append(StepResult(step=step, outcome=outcome))
            self.ctx.record_step(
                intent=step.rationale or step.agent,
                agent_name=step.agent,
                kwargs=step.kwargs,
                outcome_success=bool(outcome.success),
                outcome_reason=outcome.reason,
            )
            if not outcome.success and stop_on_failure:
                logger.info(
                    "PlanExecutor: step %r failed (%s); aborting plan",
                    step.agent, outcome.reason,
                )
                break
        return results

    async def _snapshot(self, label: str) -> None:
        """Best-effort frame capture; never raises into the executor."""
        if self.ctx.capture is None:
            return
        try:
            frame = await self.ctx.capture.capture_frame()
            self.ctx.record_frame(frame.image, label=label)
        except Exception as e:  # noqa: BLE001
            logger.debug("executor snapshot %s failed: %s", label, e)

    async def _run_step(self, step: PlanStep) -> Outcome:
        if step.agent == "key_combo":
            return await _key_combo_runner(self.ctx, **step.kwargs)
        cls = _AGENT_CLASSES.get(step.agent)
        if cls is None:
            return Outcome(
                success=False,
                reason=f"unknown agent in plan: {step.agent!r}",
            )
        try:
            return await cls(self.ctx).run(**step.kwargs)
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent %r raised", step.agent)
            return Outcome(
                success=False,
                reason=f"{step.agent} raised: {e}",
            )


__all__ = ["PlanExecutor", "StepResult"]
