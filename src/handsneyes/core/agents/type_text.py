"""TypeAgent — send text to the target machine, with optional secret mode."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from handsneyes.core.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class TypeOutcome(Outcome):
    pass


class TypeAgent(Agent):
    """Type text via the keyboard backend.

    ``secret=True`` redacts the text from any local logs (Pi side
    already only logs length). ``submit=True`` follows up with Enter.
    """

    name = "type"

    async def run(
        self,
        *,
        text: str,
        secret: bool = False,
        submit: bool = False,
        post_settle: float = 0.6,
        warmup: bool = True,
    ) -> TypeOutcome:
        if self.ctx.keyboard is None:
            return TypeOutcome(
                success=False, reason="no keyboard in context",
            )
        if not text and not submit:
            return TypeOutcome(
                success=False, reason="empty text and submit=False",
            )
        try:
            if text:
                try:
                    await self.ctx.keyboard.send_text(
                        text, secret=secret, warmup=warmup,
                    )
                except TypeError:
                    # Backend doesn't support warmup kwarg yet.
                    await self.ctx.keyboard.send_text(text, secret=secret)
                await asyncio.sleep(post_settle)
            if submit:
                await self.ctx.keyboard.send_keystroke("Enter")
        except TypeError:
            # Older keyboard backends without secret kwarg.
            await self.ctx.keyboard.send_text(text)
            await asyncio.sleep(post_settle)
            if submit:
                await self.ctx.keyboard.send_keystroke("Enter")
        except Exception as e:
            logger.warning("TypeAgent send failed: %s", e)
            return TypeOutcome(
                success=False, reason=f"send failed: {e}",
            )
        if secret:
            return TypeOutcome(
                success=True,
                reason=f"sent (length={len(text)}, redacted), submit={submit}",
            )
        return TypeOutcome(
            success=True,
            reason=f"sent {text[:40]!r}{'...' if len(text) > 40 else ''}, submit={submit}",
        )
