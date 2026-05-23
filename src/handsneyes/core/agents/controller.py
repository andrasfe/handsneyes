"""Controller — rule planner + thin ControllerAgent over PlanExecutor.

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

    # ── navigate (URL-like) ────────────────────────────────────────
    # "go to <url>" / "open <url>" / "navigate to <url>" / "browse to <url>"
    # / a bare URL like "https://reddit.com/r/LocalLLaMA". The host part
    # is what disambiguates this from a generic "open the calculator" —
    # require a dotted domain or an explicit scheme.
    nav_m = re.match(
        r"^\s*(?:go\s+to|open|navigate\s+to|browse\s+to|visit)\s+(.+)$",
        intent,
        re.IGNORECASE,
    )
    url_candidate: str | None = None
    if nav_m:
        candidate = nav_m.group(1).strip().rstrip(".,;:")
        if "." in candidate or "://" in candidate:
            url_candidate = candidate
    elif re.match(r"^\s*https?://\S+\s*$", intent, re.IGNORECASE):
        url_candidate = intent.strip()
    if url_candidate:
        if not re.match(r"^https?://", url_candidate, re.IGNORECASE):
            url_candidate = "https://" + url_candidate
        return [
            PlanStep(
                agent="navigate",
                kwargs={"url": url_candidate},
                rationale=f"navigate-keyword match → {url_candidate}",
            ),
        ]

    # ── cd / go-to-path (shell path, not a URL) ────────────────────
    # "cd ~/foo", "go to ~/foo", "navigate to /var/log" → TypeAgent
    # emitting `cd <path>` + Enter. URL nav above already claimed the
    # dotted-domain shape, so what's left here are unix paths.
    path_m = re.match(
        r"^\s*(?:cd|go\s+to|navigate\s+to)\s+(~?(?:/\S*)?|~|/\S+)\s*$",
        intent,
        re.IGNORECASE,
    )
    if path_m:
        path = path_m.group(1).strip().rstrip(".,;:")
        if path:
            return [
                PlanStep(
                    agent="type",
                    kwargs={"text": f"cd {path}", "submit": True},
                    rationale=f"cd-path match → {path!r}",
                ),
            ]

    # ── open <app> [and <rest>] ────────────────────────────────────
    # Catches "open a terminal", "open the calculator", and the
    # chained form "open a terminal and go to ~/foo" where the tail
    # is re-planned recursively (here that yields a `cd <path>` type
    # step). Runs AFTER url-nav, so "open reddit.com" still routes
    # to navigate.
    open_m = re.match(r"^\s*open\s+(.+?)\s*$", intent, re.IGNORECASE)
    if open_m:
        rest = open_m.group(1).strip()
        head, _, tail = rest.partition(" and ")
        app_name = head.strip().rstrip(".,;:")
        if app_name:
            steps: list[PlanStep] = [
                PlanStep(
                    agent="open_app",
                    kwargs={"app": app_name},
                    rationale=f"open-keyword match → {app_name!r}",
                ),
            ]
            if tail.strip():
                steps.extend(plan_intent(tail.strip()))
            return steps

    # ── click (find-target-and-click) ─────────────────────────────
    # "click <target>" → ClickAgent target= remainder.
    click_m = re.match(r"^\s*click\s+(?:on\s+)?(.+)$", intent, re.IGNORECASE)
    if click_m:
        target = click_m.group(1).strip()
        return [
            PlanStep(
                agent="click",
                kwargs={"target": target},
                rationale=f"click-keyword match → {target!r}",
            ),
        ]

    # ── focus ──────────────────────────────────────────────────────
    if re.match(r"^\s*(focus|maximi[sz]e)\b", text):
        return [
            PlanStep(
                agent="focus",
                kwargs={},
                rationale="focus-keyword match",
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


# ─── ControllerAgent shim ───────────────────────────────────────────


class ControllerAgent:
    """Compose ``plan_intent`` + :class:`PlanExecutor` into the
    one-shot entrypoint the Command Center runner expects.

    Accepts (and ignores, for API parity) the legacy kwargs the
    terminaleyes ControllerAgent took: ``no_focus``, ``vault_name``,
    ``platform``, ``allow_llm_fallback``, ``planner``, ``ml_adapter``.
    Honours ``dry_run`` (skips execution, returns the plan only).
    """

    name = "controller"

    def __init__(self, ctx: Any) -> None:  # noqa: ANN401
        self.ctx = ctx

    async def run(
        self,
        *,
        intent: str,
        no_focus: bool = False,  # noqa: ARG002
        vault_name: str | None = None,
        password: str | None = None,
        skip_verify: bool = False,
        platform: str = "linux",  # noqa: ARG002
        dry_run: bool = False,
        allow_llm_fallback: bool = True,  # noqa: ARG002
        planner: str = "auto",  # noqa: ARG002
        ml_adapter: object | None = None,  # noqa: ARG002
        **_extra: object,
    ) -> Any:  # noqa: ANN401
        from dataclasses import replace

        from handsneyes.core.agents.base import Outcome
        from handsneyes.core.agents.executor import PlanExecutor

        plan = plan_intent(intent)
        # Thread vault_name + direct password into any login step the
        # planner produced. The cc Unlock button passes one or both
        # alongside the intent, not in the intent string. Direct
        # password wins over vault lookup if both are set.
        if vault_name or password or skip_verify:
            def _inject(step: PlanStep) -> PlanStep:
                if step.agent != "login":
                    return step
                kw = dict(step.kwargs)
                if password and "password" not in kw:
                    kw["password"] = password
                if vault_name and "vault_name" not in kw:
                    kw["vault_name"] = vault_name
                if skip_verify and "verify" not in kw:
                    # LoginAgent.verify=False skips the polled visual
                    # lock-screen check. Operator's eyes-on-target.
                    kw["verify"] = False
                return replace(step, kwargs=kw)
            plan = [_inject(s) for s in plan]
        # Redact secrets when stringifying — the plan strings show up
        # in /api/runs responses and the cc history pane. Never leak
        # passwords through the public surface.
        _SECRETS = frozenset({"password", "vault_passphrase"})

        def _redact(kw: dict[str, Any]) -> dict[str, Any]:
            return {
                k: ("<redacted>" if k in _SECRETS and v else v)
                for k, v in kw.items()
            }

        plan_strs = [
            f"{step.agent}({_redact(step.kwargs)!r})"
            for step in plan
        ]
        if not plan:
            return Outcome(
                success=False,
                reason=(
                    "no rule matched this intent — rule-planner only; "
                    "LLM fallback deferred"
                ),
                data={"plan": []},
            )
        if dry_run:
            return Outcome(
                success=True,
                reason=f"planned {len(plan)} step(s) (dry-run)",
                data={"plan": plan_strs, "dry_run": True},
            )
        executor = PlanExecutor(self.ctx)
        results = await executor.run(plan)
        all_ok = all(r.outcome.success for r in results)
        return Outcome(
            success=all_ok,
            reason=(
                f"{len(results)} step(s) executed; "
                f"{'all-ok' if all_ok else 'first failure aborted plan'}"
            ),
            data={
                "plan": plan_strs,
                "results": [r.as_dict() for r in results],
            },
        )
