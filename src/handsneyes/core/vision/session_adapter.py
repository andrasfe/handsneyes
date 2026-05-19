"""Session-shape adapter wrapping :class:`AgentContext` for the homer.

The visual servo homer was built against terminaleyes'
``InteractiveSession`` and reaches into private attributes
(``_capture``, ``_client``, ``_model``, ``_evaluator``,
``_executor._mouse``, ``_executor._keyboard``). This adapter exposes
those names backed by an :class:`AgentContext`, so the homer can run
inside the agent layer without modification.

Also provides a tiny :class:`_EvaluatorShim` because handsneyes
dropped the legacy ``commander.evaluator`` module — the homer only
ever calls ``_best_text_from_response`` and ``_extract_json`` on it.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from handsneyes.core.vision.calibration import MOVE_DELAY, MOVE_STEP_SIZE

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext


class ConditionEvaluator:
    """Minimal stand-in for the legacy commander.evaluator helpers.

    Exposes the two methods the homer + ported cc/factory call:
    ``_best_text_from_response`` and ``_extract_json``. Constructor
    accepts the kwargs the cc factory passes (model, base_url,
    max_tokens) and ignores them — the shim has no actual LLM client
    of its own; agents that need one carry it via AgentContext.
    """

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str = "",
        max_tokens: int = 800,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens

    @staticmethod
    def _best_text_from_response(resp: Any) -> str:  # noqa: ANN401
        if resp is None:
            return ""
        try:
            return resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed


class _ExecutorAdapter:
    """Exposes ``_mouse`` and ``_keyboard`` under the names the homer
    expects."""

    def __init__(self, ctx: AgentContext) -> None:
        self._mouse = ctx.mouse
        self._keyboard = ctx.keyboard


class SessionAdapter:
    """Wrap an :class:`AgentContext` as an interactive-session-shaped
    object the homer can consume."""

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx
        self._capture = ctx.capture
        self._client = ctx.vision_client
        self._model = ctx.vision_model
        self._evaluator = ConditionEvaluator()
        # The homer drops per-step images alongside the rest of the
        # session artefacts.
        self.output_dir = ctx.output_dir
        self._executor = _ExecutorAdapter(ctx)

    async def _ensure_client(self) -> None:
        # Vision client is created up-front by the controller; the
        # session is a passive wrapper.
        return None

    async def _send_hid_moves(self, dx: int, dy: int) -> None:
        """Chunked mouse move matching the interactive REPL's
        throttled-send rhythm.
        """
        if self._ctx.mouse is None:
            return
        rem_x, rem_y = dx, dy
        while rem_x != 0 or rem_y != 0:
            sx = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, rem_x))
            sy = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, rem_y))
            if sx != 0 or sy != 0:
                await self._ctx.mouse.move(sx, sy)
            rem_x -= sx
            rem_y -= sy
            await asyncio.sleep(MOVE_DELAY)

    async def _showui_query(
        self, b64: str, prompt: str,
    ) -> tuple[float, float] | None:
        if self._ctx.showui_query is None:
            return None
        return await self._ctx.showui_query(b64, prompt)  # type: ignore[no-any-return]
