# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commandcenter; lint cleanup deferred.
"""Run a single ControllerAgent intent at a time, with full log capture.

The runner is the only component that touches the AgentContext mid-run.
Logs and ``print`` output are piped through the LogBus so the SSE stream
gets exactly what the user would see in a terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from handsneyes.ui.log_bus import (
    LogBus, LogEvent, make_stdout_streams,
)

# A factory builds (ctx, keyboard, mouse, capture). The runner closes
# them all when the run ends. This matches the lifecycle of `handsneyes
# do` exactly — no shared resources held across runs.
ContextFactory = Callable[[], Awaitable[tuple[Any, Any, Any, Any]]]

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    run_id: str
    intent: str
    options: dict[str, Any]
    status: str = "pending"   # pending | running | succeeded | failed | error
    started_at: float | None = None
    ended_at: float | None = None
    reason: str | None = None
    plan: list[str] = field(default_factory=list)

    _SECRET_OPTS = frozenset({"password", "vault_passphrase"})

    def public(self) -> dict:
        # Redact secrets — these stay in self.options for use during
        # _execute but never leave the process through /api/runs.
        public_opts = {
            k: ("<redacted>" if k in self._SECRET_OPTS and v else v)
            for k, v in self.options.items()
        }
        return {
            "run_id": self.run_id,
            "intent": self.intent,
            "options": public_opts,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "reason": self.reason,
            "plan": self.plan,
        }


class RunnerBusy(RuntimeError):
    pass


class Runner:
    """One-at-a-time ControllerAgent runner."""

    def __init__(
        self, context_factory: ContextFactory, bus: LogBus,
    ) -> None:
        self._context_factory = context_factory
        self.bus = bus
        self._records: dict[str, RunRecord] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()
        self._active: RunRecord | None = None
        self._task: asyncio.Task | None = None

    def is_busy(self) -> bool:
        return self._active is not None

    def active(self) -> RunRecord | None:
        return self._active

    def get(self, run_id: str) -> RunRecord | None:
        return self._records.get(run_id)

    def list(self, *, limit: int = 50) -> list[RunRecord]:
        ids = self._order[-limit:]
        return [self._records[i] for i in reversed(ids)]

    async def start(
        self,
        *,
        intent: str,
        no_focus: bool = False,
        vault: str | None = None,
        password: str | None = None,
        vault_passphrase: str | None = None,
        skip_verify: bool = False,
        platform: str = "linux",
        dry_run: bool = False,
        allow_llm_fallback: bool = True,
        planner: str = "auto",
        ml_adapter: str | None = None,
    ) -> RunRecord:
        async with self._lock:
            if self._active is not None:
                raise RunnerBusy(
                    f"another run is in progress: {self._active.run_id}"
                )
            run_id = uuid.uuid4().hex[:12]
            record = RunRecord(
                run_id=run_id,
                intent=intent,
                options={
                    "no_focus": no_focus, "vault": vault,
                    "password": password,
                    "vault_passphrase": vault_passphrase,
                    "skip_verify": skip_verify,
                    "platform": platform, "dry_run": dry_run,
                    "allow_llm_fallback": allow_llm_fallback,
                    "planner": planner, "ml_adapter": ml_adapter,
                },
                status="running",
                started_at=time.time(),
            )
            self._records[run_id] = record
            self._order.append(run_id)
            self._active = record
            self._task = asyncio.create_task(
                self._execute(record), name=f"run-{run_id}",
            )
            return record

    async def _execute(self, record: RunRecord) -> None:
        # Late import: ControllerAgent pulls heavy deps. Guard against
        # import-time failure so the runner can never leave the record
        # stuck in "running" (the bug the verbatim port shipped with).
        try:
            from handsneyes.core.agents.controller import ControllerAgent
        except Exception as e:
            record.status = "error"
            record.reason = f"controller import failed: {e}"
            record.ended_at = time.time()
            self.bus.publish(LogEvent(
                ts=time.time(), level="ERROR", source="system",
                msg=f"! run {record.run_id} {record.reason}",
                run_id=record.run_id,
            ))
            self.bus.close_run(record.run_id)
            self._active = None
            self._task = None
            return

        bus = self.bus
        run_id = record.run_id
        bus.publish(LogEvent(
            ts=time.time(), level="INFO", source="system",
            msg=f"▶ run {run_id}: {record.intent!r}", run_id=run_id,
        ))
        out_stream, err_stream = make_stdout_streams(bus)
        ctx = keyboard = mouse = capture = None
        try:
            with bus.active_run(run_id), \
                 contextlib.redirect_stdout(out_stream), \
                 contextlib.redirect_stderr(err_stream):
                ctx, keyboard, mouse, capture = await self._context_factory()
                # Per-request vault passphrase override: lets the UI's
                # Unlock button supply the passphrase inline instead
                # of requiring an env var. Built fresh per run so
                # passphrases never leak into the next request.
                vp = record.options.get("vault_passphrase")
                if vp:
                    try:
                        from handsneyes.core.vault import Vault
                        ctx.vault = Vault(vp)
                    except Exception as e:
                        logger.warning(
                            "vault_passphrase rejected: %s", e,
                        )
                agent = ControllerAgent(ctx)
                outcome = await agent.run(
                    intent=record.intent,
                    no_focus=record.options["no_focus"],
                    vault_name=record.options["vault"],
                    password=record.options.get("password"),
                    skip_verify=record.options.get("skip_verify", False),
                    platform=record.options["platform"],
                    dry_run=record.options["dry_run"],
                    allow_llm_fallback=record.options["allow_llm_fallback"],
                    planner=record.options.get("planner", "auto"),
                    ml_adapter=record.options.get("ml_adapter"),
                )
            record.status = "succeeded" if bool(outcome) else "failed"
            record.reason = outcome.reason
            plan = (getattr(outcome, "data", {}) or {}).get("plan") or []
            if isinstance(plan, list):
                record.plan = [str(s) for s in plan]
        except Exception as e:
            logger.exception("Controller crashed in run %s", run_id)
            record.status = "error"
            record.reason = f"{type(e).__name__}: {e}"
        finally:
            # Tear down per-run resources, just like `handsneyes do`.
            if capture is not None:
                try:
                    await capture.close()
                except Exception:
                    logger.exception("capture.close failed")
            if keyboard is not None:
                try:
                    await keyboard.disconnect()
                except Exception:
                    logger.exception("keyboard.disconnect failed")
            if mouse is not None:
                try:
                    await mouse.disconnect()
                except Exception:
                    logger.exception("mouse.disconnect failed")
            record.ended_at = time.time()
            mark = {"succeeded": "✓", "failed": "✗", "error": "!"}.get(
                record.status, "?",
            )
            bus.publish(LogEvent(
                ts=time.time(), level="INFO", source="system",
                msg=f"{mark} run {run_id} {record.status}: {record.reason}",
                run_id=run_id,
            ))
            bus.close_run(run_id)
            self._active = None
            self._task = None
