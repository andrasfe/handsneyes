# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commandcenter; lint cleanup deferred.
"""Indexes PNG frames written by agents into the per-session output dir.

The store is the single source of truth for what the UI shows. It watches
the agent layer's output directory on a poll loop, assigns a monotonic
id to every new image, and serves the bytes on demand. A FIFO cap bounds
the in-memory index — the underlying files are left on disk untouched.

Default watch dir mirrors the agent layer's default
(``~/.local/share/handsneyes/runs/``) and can be overridden by setting
``HANDSNEYES_OUTPUT_DIR``. Each per-run subdirectory of the watch dir
is treated as one "run" — its name surfaces in :class:`FrameMeta.run_id`
so the UI can correlate frames to a runner record.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _default_watch_dir() -> Path:
    """Resolve the watch dir from env or fall back to the agent default."""
    env = os.environ.get("HANDSNEYES_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path.home() / ".local" / "share" / "handsneyes" / "runs"
    ).resolve()


DEFAULT_WATCH_DIR = _default_watch_dir()
DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_MAX_FRAMES = 500
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class FrameMeta:
    id: int                 # mtime in nanoseconds — monotonic-ish, unique
    ts: float               # mtime as unix seconds (for display)
    run_id: str             # name of the run dir, e.g. "062115_vs"
    filename: str           # basename within the run dir
    path: str               # absolute path on disk

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        # JSON.parse in JS rounds 64-bit nanosecond ids to the
        # nearest representable double (last 3-4 digits silently
        # zeroed). Send id as a string so the round-trip preserves
        # exact bits — FastAPI's path handler parses it back to int.
        d["id"] = str(d["id"])
        return d


class FrameStore:
    """In-memory index of frames on disk. Polls a watch directory."""

    def __init__(
        self,
        watch_dir: Path = DEFAULT_WATCH_DIR,
        max_frames: int = DEFAULT_MAX_FRAMES,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        delete_on_evict: bool = True,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.max_frames = max_frames
        self.poll_interval = poll_interval
        # When True, unlinking the file on disk follows the in-memory
        # eviction. Without this, the watch_dir grows unbounded across
        # sessions (the in-memory deque caps at max_frames, but on-disk
        # PNGs accumulated forever — quickly fills up dev macs).
        # The training pipeline reads history.jsonl, not PNGs, so the
        # only cost is replayability of stale runs.
        self.delete_on_evict = delete_on_evict
        self._frames: deque[FrameMeta] = deque(maxlen=max_frames)
        self._by_id: dict[int, FrameMeta] = {}
        self._seen_paths: set[str] = set()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._latest_id: int | None = None
        self._update_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        # One-shot startup cleanup: if disk has more than max_frames
        # PNGs across all run dirs (typical after long-running cc with
        # many click_at sessions), trim down to the newest max_frames
        # BEFORE the initial scan. Without this, an old run from
        # weeks ago would leave thousands of PNGs that the deque can't
        # see but the disk still carries.
        if self.delete_on_evict:
            self._cleanup_disk_to_max()
        await self._scan_once(initial=True)
        self._task = asyncio.create_task(self._run(), name="frame-watcher")

    def _cleanup_disk_to_max(self) -> None:
        """Delete image files beyond max_frames newest, across all run
        subdirs of watch_dir. Best-effort: a file we can't stat or
        delete is logged at debug and skipped."""
        all_files: list[tuple[float, str]] = []
        try:
            for run_entry in os.scandir(self.watch_dir):
                if not run_entry.is_dir():
                    continue
                try:
                    inner = os.scandir(run_entry.path)
                except OSError:
                    continue
                for f in inner:
                    if not f.is_file():
                        continue
                    if Path(f.path).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    try:
                        all_files.append((f.stat().st_mtime, f.path))
                    except OSError:
                        continue
        except OSError as e:
            logger.debug("frame_store cleanup scan failed: %s", e)
            return
        if len(all_files) <= self.max_frames:
            return
        all_files.sort(reverse=True)  # newest first
        to_delete = all_files[self.max_frames:]
        ok = 0
        for _, path in to_delete:
            try:
                os.unlink(path)
                ok += 1
            except OSError as e:
                logger.debug("could not delete %s: %s", path, e)
        logger.info(
            "frame_store: startup cleanup removed %d/%d old frame "
            "PNG(s); kept newest %d",
            ok, len(to_delete), self.max_frames,
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._scan_once()
            except Exception:
                logger.exception("frame_store scan failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                pass

    async def _scan_once(self, *, initial: bool = False) -> None:
        added: list[FrameMeta] = []
        try:
            run_dirs = [
                e for e in os.scandir(self.watch_dir) if e.is_dir()
            ]
        except FileNotFoundError:
            return
        for run_entry in run_dirs:
            try:
                file_entries = list(os.scandir(run_entry.path))
            except FileNotFoundError:
                continue
            for f in file_entries:
                if not f.is_file():
                    continue
                p = f.path
                if p in self._seen_paths:
                    continue
                if Path(p).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    st = f.stat()
                except FileNotFoundError:
                    continue
                meta = FrameMeta(
                    id=st.st_mtime_ns,
                    ts=st.st_mtime,
                    run_id=run_entry.name,
                    filename=f.name,
                    path=p,
                )
                added.append(meta)
                self._seen_paths.add(p)
        if not added:
            return
        added.sort(key=lambda m: m.id)
        # Disambiguate ids that collide (rare, same-ns mtimes).
        async with self._lock:
            for m in added:
                if m.id in self._by_id:
                    m = FrameMeta(
                        id=m.id + 1, ts=m.ts, run_id=m.run_id,
                        filename=m.filename, path=m.path,
                    )
                # Maintain FIFO: deque.maxlen drops left automatically.
                if (
                    len(self._frames) == self.max_frames
                    and self._frames
                ):
                    evicted = self._frames[0]
                    self._by_id.pop(evicted.id, None)
                    self._seen_paths.discard(evicted.path)
                    if self.delete_on_evict:
                        try:
                            os.unlink(evicted.path)
                        except OSError as e:
                            logger.debug(
                                "frame_store: could not unlink %s: %s",
                                evicted.path, e,
                            )
                self._frames.append(m)
                self._by_id[m.id] = m
                self._latest_id = m.id
        if not initial:
            logger.debug("frame_store: %d new frame(s)", len(added))
        # Wake any awaiters (latest-frame long-poll).
        self._update_event.set()
        self._update_event.clear()

    # ── read API ──────────────────────────────────────────────────

    def list(
        self, *, limit: int | None = None, before: int | None = None,
    ) -> list[FrameMeta]:
        """Newest-first listing. ``before`` returns ids strictly less than it."""
        items = list(self._frames)
        items.reverse()
        if before is not None:
            items = [m for m in items if m.id < before]
        if limit is not None:
            items = items[:limit]
        return items

    def latest(self) -> FrameMeta | None:
        if self._latest_id is None:
            return None
        return self._by_id.get(self._latest_id)

    def get(self, frame_id: int) -> FrameMeta | None:
        return self._by_id.get(frame_id)

    def neighbours(self, frame_id: int) -> tuple[int | None, int | None]:
        """Return (prev_id, next_id) by index order."""
        ids = [m.id for m in self._frames]
        try:
            i = ids.index(frame_id)
        except ValueError:
            return None, None
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i + 1 < len(ids) else None
        return prev_id, next_id

    def count(self) -> int:
        return len(self._frames)

    async def wait_for_update(self, current_id: int | None) -> FrameMeta | None:
        """Block until the latest id changes (or we're already ahead)."""
        if self._latest_id is not None and self._latest_id != current_id:
            return self.latest()
        try:
            await asyncio.wait_for(self._update_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            return None
        return self.latest()
