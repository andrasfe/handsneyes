"""ControllerAgent — minimal rule planner for Phase A.

Decomposes free-form intents into a typed list of ``PlanStep`` rows.
Phase A scope: enough rules to demonstrate end-to-end dispatching
through the agent + platform-adapter layer. The full terminaleyes
controller (~2400 LoC with LLM-fallback, scene-aware caching, etc.)
lands in Phase B alongside the click/navigate workflow agents.

Rule whitelist (case-insensitive contains-match unless noted):
  - "scroll down" / "scroll up"   → ScrollAgent
  - "type ..." / "send keystrokes" → TypeAgent
  - "login" / "unlock" / "sign in" → LoginAgent
  - "lock"                        → key combo super+L (HID via platform)
  - everything else               → unresolved (caller decides)

This rule layer never *executes* anything — it returns a plan. The
CLI's dry-run mode prints the plan and exits; non-dry-run mode would
hand the plan to the agent layer, but Phase A's smoke goal stops at
the plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    """One step in a controller's plan."""

    agent: str  # logical agent name, e.g. "scroll", "type", "login"
    kwargs: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "kwargs": self.kwargs,
            "rationale": self.rationale,
        }


_SCROLL_AMOUNT = re.compile(r"\bby\s+(\d+)\b|\b(\d+)\b")


def _parse_scroll_amount(intent: str, default: int = 4) -> int:
    m = _SCROLL_AMOUNT.search(intent)
    if not m:
        return default
    g = m.group(1) or m.group(2)
    try:
        n = int(g)
        return max(1, min(n, 30))
    except (TypeError, ValueError):
        return default


def plan_intent(intent: str) -> list[PlanStep]:
    """Translate a free-form intent into a sequence of plan steps.

    Returns an empty list when no rule matches — caller decides whether
    to refuse, fall back to an LLM planner, or surface the gap.
    """
    text = intent.strip().lower()
    if not text:
        return []

    # ── login / unlock ─────────────────────────────────────────────
    if any(kw in text for kw in ("login", "log in", "unlock", "sign in")):
        m = re.search(r"--vault\s+(\S+)", intent)
        kwargs: dict[str, Any] = {}
        if m:
            kwargs["vault_name"] = m.group(1)
        return [
            PlanStep(
                agent="login",
                kwargs=kwargs,
                rationale="login-keyword match",
            ),
        ]

    # ── scroll ─────────────────────────────────────────────────────
    if "scroll" in text:
        direction = "up" if "up" in text else "down"
        amount = _parse_scroll_amount(intent)
        return [
            PlanStep(
                agent="scroll",
                kwargs={"direction": direction, "amount": amount},
                rationale=f"scroll-keyword match ({direction}×{amount})",
            ),
        ]

    # ── lock ────────────────────────────────────────────────────────
    if re.match(r"^\s*lock\b", text):
        return [
            PlanStep(
                agent="key_combo",
                kwargs={"modifiers": ["super"], "key": "l"},
                rationale="lock screen → Super+L",
            ),
        ]

    # ── type ────────────────────────────────────────────────────────
    # Crude: "type X" → send X. Anything richer (URL, keystrokes,
    # multi-line) is Phase B controller's job.
    m = re.match(r"^\s*(type|send)\s+(.+)$", intent, re.IGNORECASE)
    if m:
        payload = m.group(2).strip()
        # Strip surrounding quotes if user wrote: type "hello there"
        if (payload.startswith('"') and payload.endswith('"')) or (
            payload.startswith("'") and payload.endswith("'")
        ):
            payload = payload[1:-1]
        return [
            PlanStep(
                agent="type",
                kwargs={"text": payload, "submit": False},
                rationale="type-keyword match",
            ),
        ]

    return []
