# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commandcenter; lint cleanup deferred.
"""Pub/sub log bus for the command center.

A logging.Handler captures records from the ``handsneyes`` namespace and
fans them out to per-run + global asyncio queues. SSE streams subscribe to
those queues. ``stdout``/``stderr`` from the controller (which uses
``print``) is also redirected through the bus while a run is active.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

LOGGER_NAME = "handsneyes"
SENTINEL: object = object()


@dataclass
class LogEvent:
    ts: float
    level: str
    source: str       # "logger" | "stdout" | "stderr" | "system"
    msg: str
    run_id: str | None

    def public(self) -> dict:
        return {
            "ts": self.ts, "level": self.level,
            "source": self.source, "msg": self.msg,
            "run_id": self.run_id,
        }


class LogBus:
    """In-process pub/sub. Per-run queues + a global queue."""

    def __init__(self, max_global: int = 2000) -> None:
        self._global: list[LogEvent] = []
        self._max_global = max_global
        self._global_subs: set[asyncio.Queue] = set()
        self._run_subs: dict[str, set[asyncio.Queue]] = {}
        self._run_buf: dict[str, list[LogEvent]] = {}
        self._closed_runs: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_run_id: str | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ── publication ───────────────────────────────────────────────

    def publish(self, ev: LogEvent) -> None:
        # Append to global ring.
        self._global.append(ev)
        if len(self._global) > self._max_global:
            del self._global[: len(self._global) - self._max_global]
        # Append to per-run buffer (so late subscribers get history).
        if ev.run_id is not None:
            buf = self._run_buf.setdefault(ev.run_id, [])
            buf.append(ev)
        # Fan out.
        for q in list(self._global_subs):
            self._safe_put(q, ev)
        if ev.run_id is not None:
            for q in list(self._run_subs.get(ev.run_id, ())):
                self._safe_put(q, ev)

    def _safe_put(self, q: asyncio.Queue, item: object) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def close_run(self, run_id: str) -> None:
        self._closed_runs.add(run_id)
        for q in list(self._run_subs.get(run_id, ())):
            self._safe_put(q, SENTINEL)

    # ── subscription ──────────────────────────────────────────────

    async def subscribe_run(
        self, run_id: str, *, replay: bool = True,
    ) -> AsyncIterator[LogEvent]:
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self._run_subs.setdefault(run_id, set()).add(q)
        try:
            if replay:
                for ev in list(self._run_buf.get(run_id, ())):
                    yield ev
            if run_id in self._closed_runs and q.empty():
                return
            while True:
                item = await q.get()
                if item is SENTINEL:
                    return
                yield item  # type: ignore[misc]
        finally:
            self._run_subs.get(run_id, set()).discard(q)

    async def subscribe_global(
        self, *, replay_tail: int = 200,
    ) -> AsyncIterator[LogEvent]:
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self._global_subs.add(q)
        try:
            for ev in self._global[-replay_tail:]:
                yield ev
            while True:
                item = await q.get()
                if item is SENTINEL:
                    return
                yield item  # type: ignore[misc]
        finally:
            self._global_subs.discard(q)

    # ── current-run association ───────────────────────────────────

    @contextmanager
    def active_run(self, run_id: str) -> Iterator[None]:
        prev = self._current_run_id
        self._current_run_id = run_id
        try:
            yield
        finally:
            self._current_run_id = prev

    def current_run_id(self) -> str | None:
        return self._current_run_id


class _BusLoggingHandler(logging.Handler):
    """logging.Handler that pushes records into a LogBus."""

    def __init__(self, bus: LogBus) -> None:
        super().__init__()
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        ev = LogEvent(
            ts=time.time(),
            level=record.levelname,
            source="logger",
            msg=msg,
            run_id=self.bus.current_run_id(),
        )
        self.bus.publish(ev)


class _BusStream(io.TextIOBase):
    """File-like that turns writes into LogEvents on a bus."""

    def __init__(self, bus: LogBus, source: str) -> None:
        super().__init__()
        self.bus = bus
        self.source = source
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self.bus.publish(LogEvent(
                    ts=time.time(),
                    level="INFO",
                    source=self.source,
                    msg=line,
                    run_id=self.bus.current_run_id(),
                ))
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self.bus.publish(LogEvent(
                ts=time.time(), level="INFO",
                source=self.source, msg=self._buf,
                run_id=self.bus.current_run_id(),
            ))
            self._buf = ""


def install_logging(bus: LogBus, level: int = logging.INFO) -> _BusLoggingHandler:
    """Attach a bus handler to the handsneyes logger. Returns the handler."""
    handler = _BusLoggingHandler(bus)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)
    return handler


def make_stdout_streams(bus: LogBus) -> tuple[_BusStream, _BusStream]:
    return _BusStream(bus, "stdout"), _BusStream(bus, "stderr")
