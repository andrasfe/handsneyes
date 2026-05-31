# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commandcenter; lint cleanup deferred.
"""FastAPI app for the command center.

Endpoints:
  GET  /                         -> static index.html
  GET  /api/frames               -> list newest-first {limit, before}
  GET  /api/frames/latest        -> meta of newest frame  (optional ?wait=1)
  GET  /api/frames/{id}          -> JPEG/PNG bytes
  GET  /api/frames/{id}/neighbours
  POST /api/run                  -> start a ControllerAgent run
  GET  /api/runs                 -> list recent runs
  GET  /api/runs/{id}            -> single run record
  GET  /api/runs/{id}/logs       -> SSE stream of LogEvents
  GET  /api/logs                 -> SSE stream of all logs
  GET  /api/state                -> {busy, latest_id, run?}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from handsneyes.ui.frame_store import FrameStore
from handsneyes.ui.log_bus import LogBus, LogEvent, install_logging
from handsneyes.ui.runner import (
    ContextFactory, Runner, RunnerBusy,
)

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    intent: str = Field(min_length=1)
    no_focus: bool = False
    vault: str | None = None
    # Direct unlock password (bypasses the vault chain). The cc UI's
    # Unlock button uses this when the operator chooses "type it once"
    # over "set up the vault" — no env var or terminal-side getpass
    # required. Server treats it as opaque; LoginAgent receives it
    # via the controller's threading and sends it through the
    # secret=True keyboard path.
    password: str | None = None
    # Vault passphrase override. When set, the runner builds a Vault
    # at request time and feeds it to the AgentContext, so LoginAgent
    # can look up `vault` entries without a pre-set env passphrase.
    vault_passphrase: str | None = None
    # Skip the LoginAgent's visual lock-screen verify and just type
    # the password. Useful when the webcam is unavailable / fooled
    # (e.g. the macOS SMPTE-bars placeholder makes the vision LLM
    # hallucinate a yes). Operator's eyes-on-target — they're
    # asserting the target IS on a lock screen.
    skip_verify: bool = False
    platform: str = "linux"
    dry_run: bool = False
    allow_llm_fallback: bool = True
    planner: str = "auto"           # "auto" | "ml" | "rules"
    ml_adapter: str | None = None   # required when planner == "ml"


class ScheduleCreateRequest(BaseModel):
    """Recurring controller intent. Each tick fires the same RunRequest
    shape; a tick where the runner is busy is silently skipped (a tune
    or earlier scheduled job wins). Skipped ticks don't shift the cadence
    — the next fire is still N minutes after the previous SCHEDULED tick,
    not after the skipped one."""
    intent: str = Field(min_length=1)
    interval_minutes: float = Field(gt=0.0, le=24 * 60.0)
    # Same options as RunRequest. Snapshot of the chat-form controls at
    # the moment of creation; later changes to the UI don't affect
    # already-scheduled jobs.
    no_focus: bool = False
    vault: str | None = None
    password: str | None = None
    vault_passphrase: str | None = None
    skip_verify: bool = False
    platform: str = "linux"
    dry_run: bool = False
    allow_llm_fallback: bool = True
    planner: str = "auto"
    ml_adapter: str | None = None
    # When True, the first fire happens immediately on create (and the
    # next is +interval_minutes from then). False = wait one full
    # interval before the first fire.
    fire_immediately: bool = False


class ScheduleCancelRequest(BaseModel):
    id: str = Field(min_length=1)


class PointerAccelScaleRequest(BaseModel):
    """Per-axis multiplier applied to the pointer-accel model's HID
    output, to bridge a dev / target effective-resolution gap without
    retraining. Range chosen so the user can dial the cursor in
    interactively from the cc UI."""
    scale_x: float = Field(ge=0.05, le=4.0)
    scale_y: float = Field(ge=0.05, le=4.0)


class CaptureSourceRequest(BaseModel):
    self_capture: bool


class MouseClickAtRequest(BaseModel):
    x_pct: float = Field(ge=0.0, le=1.0)
    y_pct: float = Field(ge=0.0, le=1.0)
    button: str = Field(default="left", pattern="^(left|right|middle)$")
    # ``count=2`` → home to the pixel and fire a double-click. The
    # homer always fires one click as part of landing; for count > 1
    # we send (count - 1) extra in-place clicks after the home is
    # reported successful.
    count: int = Field(default=1, ge=1, le=3)
    # Optional overrides; defaults come from settings.commander.
    screen_width: int | None = Field(default=None, gt=0)
    screen_height: int | None = Field(default=None, gt=0)


class MouseDragRequest(BaseModel):
    from_x_pct: float = Field(ge=0.0, le=1.0)
    from_y_pct: float = Field(ge=0.0, le=1.0)
    to_x_pct: float = Field(ge=0.0, le=1.0)
    to_y_pct: float = Field(ge=0.0, le=1.0)
    button: str = Field(default="left", pattern="^(left|right|middle)$")


class MouseClickRequest(BaseModel):
    button: str = Field(default="left", pattern="^(left|right|middle)$")
    # ``count=2`` → double-click (two clicks within the OS's
    # double-click threshold). Capped at 3 because more than that is
    # almost never useful and would just be the BT HID timing budget.
    count: int = Field(default=1, ge=1, le=3)


class MouseMoveRequest(BaseModel):
    dx: int = Field(ge=-127, le=127)
    dy: int = Field(ge=-127, le=127)


class MouseScrollRequest(BaseModel):
    """Wheel-tick scroll. ``amount`` is signed in mouse-wheel units —
    positive for "scroll down/away from user", negative for "up".
    Matches the Pi-side ``mouse.scroll(amount)`` convention.

    ``x_pct`` / ``y_pct`` are optional and only carried for telemetry/
    snapshot labelling today; the scroll is applied at the target's
    current cursor position. Moving the target cursor to the operator's
    hover position before scrolling is a future enhancement, gated on
    a faster open-loop home path (the current homer is too slow for
    per-wheel-event latency).
    """
    amount: int = Field(ge=-30, le=30)
    x_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    y_pct: float | None = Field(default=None, ge=0.0, le=1.0)


class KeyboardTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    warmup: bool = False
    # When typing a password / secret into a host field, set secret=True
    # so the dev-side log records only the length (the Pi side already
    # only logs the length). The cc UI's "Type Secret" button sets this.
    secret: bool = False
    append_enter: bool = False


class KeyboardKeyRequest(BaseModel):
    key: str = Field(min_length=1, max_length=32)
    modifiers: list[str] = Field(default_factory=list)


class PasteFileRequest(BaseModel):
    # 50 KB cap: BT HID throughput on this stack is roughly
    # 30–50 chars/sec; bigger payloads turn into multi-minute waits
    # with very low odds of clean OCR verification.
    content: str = Field(min_length=1, max_length=50_000)
    filename: str = Field(default="cc_paste.txt", max_length=128)
    path: str = Field(default="/tmp/cc_paste.txt", max_length=256)
    platform: str = Field(default="macos", pattern="^(macos|linux)$")
    maximize: bool = True
    verify: bool = True
    # Optional pager-driven body readback. SHA-256 (under ``verify``)
    # is the *cryptographic* identity check; this is for visual /
    # body-level confirmation — drive ``more PATH`` and OCR each
    # page so the operator can see the file scroll past via webcam.
    # Disabled by default because each page costs ~1 s of dwell.
    body_readback: bool = False


class VaultCreateRequest(BaseModel):
    # New master passphrase for the fresh vault.
    passphrase: str = Field(min_length=1, max_length=512)
    # Entry name + value to seed the new vault with. Typically the
    # operator creates the vault and the unlock entry in one shot.
    entry_name: str = Field(min_length=1, max_length=128)
    entry_value: str = Field(min_length=1, max_length=4096)
    # Refuse if a vault file already exists, unless overwrite=True.
    # The cc UI's "Create new vault" tab sets overwrite=True after
    # showing a confirmation; programmatic callers should default
    # to the safe behaviour.
    overwrite: bool = False


class VaultAddRequest(BaseModel):
    # Existing master passphrase. Required to decrypt the vault and
    # then re-encrypt it with the new entry added.
    passphrase: str = Field(min_length=1, max_length=512)
    # New entry name + value to add. If an entry with this name
    # already exists in the vault it is REPLACED (same semantics as
    # `handsneyes vault add NAME` from the CLI).
    entry_name: str = Field(min_length=1, max_length=128)
    entry_value: str = Field(min_length=1, max_length=4096)


class VaultUnlockSessionRequest(BaseModel):
    # Master passphrase for the vault. Cached in app.state for the
    # cc process lifetime; lets later /api/keyboard/from-vault calls
    # type entries without an external caller needing the passphrase
    # or the entry value. Cleared on /api/vault/lock-session and on
    # cc shutdown.
    passphrase: str = Field(min_length=1, max_length=512)


class KeyboardFromVaultRequest(BaseModel):
    # Vault entry name (e.g. "desktop") whose value should be typed
    # at the host's currently-focused field. The value is NEVER
    # returned in the response or logged, only its length.
    entry: str = Field(min_length=1, max_length=128)
    # Press Enter after typing the value — useful for sudo prompts
    # and password fields that submit on Enter.
    append_enter: bool = True


class SyncTextRequest(BaseModel):
    # ROI centred on (x_pct, y_pct). Defaults to the last click_at
    # position so just clicking into a text field is enough setup.
    x_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    y_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    # Half-height of the cropped band as a fraction of frame height.
    # Default ±3% (so 6% total) covers a normal text input at typical
    # UI density; bump up for wrapped textareas.
    band_pct: float = Field(default=0.03, ge=0.005, le=0.2)
    # Override the OCR model. Default uses nanonets-ocr-s (small,
    # dedicated OCR model — much faster + more accurate on webcam
    # captures than the general-purpose multimodal model).
    model: str | None = None


def _content_type_for(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "application/octet-stream"


def _sse(event: str | None, data: Any) -> bytes:
    payload = json.dumps(data) if not isinstance(data, str) else data
    out = []
    if event:
        out.append(f"event: {event}")
    for line in payload.splitlines() or [""]:
        out.append(f"data: {line}")
    out.append("")
    out.append("")
    return ("\n".join(out)).encode()


def create_app(
    context_factory: ContextFactory,
    *,
    frame_store: FrameStore | None = None,
    bus: LogBus | None = None,
    settings: Any = None,
    active_platform: str = "linux_gnome",
    runtime_state: dict | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``context_factory`` is awaited at the start of each run to build a
    fresh AgentContext (mouse/keyboard/capture/etc). The runner closes
    those resources when the run ends — so the webcam is only held for
    the duration of an actual run, not while the server is idle.
    """
    store = frame_store or FrameStore()
    bus = bus or LogBus()
    install_logging(bus)
    runner = Runner(context_factory, bus)

    app = FastAPI(title="handsneyes command center")
    app.state.store = store
    app.state.bus = bus
    app.state.runner = runner
    # The active target's platform name (e.g. "linux_gnome", "macos").
    # Threaded through tune / rollback / training-state so each OS
    # gets its own model file, Previous slot, and sample counter —
    # macOS clicks don't poison the Ubuntu retrain corpus and vice
    # versa. Set from the CLI's chosen target in handsneyes/cli.py.
    app.state.active_platform = active_platform
    # Cross-cutting runtime toggles the UI can flip without a cc
    # restart. Shared by reference with make_target_context_factory
    # so the factory sees updates immediately on the next /api/run.
    app.state.runtime_state = runtime_state if runtime_state is not None else {}

    # Serializes manual-control webcam captures so a click and a
    # background follow-up snapshot don't fight over the device.
    _manual_capture_lock = asyncio.Lock()

    # Single mutex around every manual mouse action (click_at,
    # click, move, scroll, plus post-action snapshot work). Two
    # concurrent HID reports over BT cause genuinely undefined
    # behaviour — at best the second wins and the first is lost,
    # at worst the Pi rejects both. Cheap to acquire; the cc UI
    # already enforces a single-in-flight discipline at the JS
    # level, this is the belt-and-braces guarantee on the server.
    _manual_mouse_lock = asyncio.Lock()

    # Cache of the last (x_pct, y_pct) the cursor was visually
    # homed to. /api/mouse/scroll skips a fresh home when the
    # operator hovers within tolerance of the cached position, so
    # a continuous scroll gesture re-uses the homing cost.
    app.state.last_scroll_home_xy = None
    # Cache for the no-slam follow-up path. After every successful
    # click_at the cursor is known to be at (req.x_pct, req.y_pct)
    # on the host. Subsequent click_at requests within
    # ``NO_SLAM_CACHE_TTL_S`` reuse that position as the starting
    # cursor and skip the slam-to-corner + oscillation detection
    # that would otherwise dismiss any open menu / modal / popover.
    # Invalidated by any other mouse action.
    app.state.last_click_xy_at = None  # tuple[(x,y), epoch] | None
    # Cache for the homer's learned pct-per-HID ratios. Each
    # click_at creates a fresh VisualServoHomer, so without this
    # cache the homer re-discovers the host's pointer-accel curve
    # from scratch every click (DEFAULT seed is typically 5–10×
    # off the real ratio for screen-share remote targets). With
    # the cache, the first click calibrates and every later click
    # within the session starts already-calibrated. tuple[
    # (ratio_x, ratio_y), epoch] | None — same TTL as
    # last_click_xy_at since the ratio is host-specific and a
    # long gap probably means the target/display config changed.
    app.state.last_homer_ratio_at = None

    # Homer-retrain state. n_trajectories_since_train is incremented
    # after every successful click_at — each click writes a fresh
    # history.jsonl which is exactly the kind of row build_*_dataset
    # consumes. is_retraining is True while a retrain subprocess
    # runs (we lock the endpoint behind this so two concurrent
    # retrains can't fight over the checkpoint dir).
    app.state.n_trajectories_since_train = 0
    app.state.last_retrain = None  # dict | None — most recent verdict
    app.state.is_retraining = False
    # Cached vault passphrase for the session. Once an operator
    # unlocks the vault via /api/vault/unlock-session, later
    # /api/keyboard/from-vault calls can type stored entries without
    # the caller needing to know either the passphrase or the value.
    # Lives in process memory only — gone on cc restart.
    app.state.vault_passphrase = None
    # Recurring scheduler. {job_id -> dict of metadata + asyncio.Task}.
    # In-memory only — jobs are forgotten on cc restart. The "Tune"
    # use case for the scheduler is "fire this intent every N minutes
    # while I work", which lives the same lifetime as a cc session.
    app.state.schedules: dict[str, dict] = {}

    @app.on_event("startup")
    async def _on_startup() -> None:
        bus.bind_loop(asyncio.get_event_loop())
        await store.start()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await store.stop()

    # ── static / index ───────────────────────────────────────────
    # No-cache headers so iterating on the SPA doesn't require the
    # user to hard-refresh after every server restart. Static files
    # are tiny — caching savings aren't worth the dev friction.
    NO_CACHE = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    if STATIC_DIR.exists():
        class _NoCacheStatic(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                for k, v in NO_CACHE.items():
                    resp.headers[k] = v
                return resp

        app.mount(
            "/static",
            _NoCacheStatic(directory=str(STATIC_DIR)),
            name="static",
        )

    @app.get("/")
    def index() -> FileResponse:
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(404, "index.html not found")
        return FileResponse(str(idx), headers=NO_CACHE)

    # ── scripts download ─────────────────────────────────────────
    # Convenience endpoint so the operator can fetch helper shell
    # scripts from the target machine with a single curl, avoiding
    # scp / USB / etc. round-trips. Only ``.sh`` files in the
    # repo's ``scripts/`` directory are exposed; path traversal is
    # blocked.
    _SCRIPTS_DIR = (
        Path(__file__).parents[3] / "scripts"
    ).resolve()

    @app.get("/scripts/{name}")
    def download_script(name: str) -> FileResponse:
        if "/" in name or ".." in name or not name.endswith(".sh"):
            raise HTTPException(400, "only .sh filenames are allowed")
        path = (_SCRIPTS_DIR / name).resolve()
        try:
            path.relative_to(_SCRIPTS_DIR)
        except ValueError:
            raise HTTPException(400, "path traversal blocked")
        if not path.is_file():
            raise HTTPException(404, f"{name} not found")
        return FileResponse(
            str(path),
            media_type="text/x-shellscript",
            filename=name,
        )

    # ── frames ────────────────────────────────────────────────────
    @app.get("/api/frames")
    def list_frames(
        limit: int = Query(50, ge=1, le=500),
        before: int | None = None,
    ) -> JSONResponse:
        items = store.list(limit=limit, before=before)
        return JSONResponse({
            "count": store.count(),
            "items": [m.public() for m in items],
        })

    @app.get("/api/frames/latest")
    async def latest_frame(
        wait: int = Query(0, ge=0, le=1),
        since: str | None = Query(None),
    ) -> JSONResponse:
        # Tolerate empty string: the SPA passes `since=` when it has
        # no known frame yet. Strict `int | None` 422s on the empty
        # string and the long-poll loop pegs the server.
        since_id: int | None = None
        if since:
            try:
                since_id = int(since)
            except ValueError:
                since_id = None
        if wait:
            meta = await store.wait_for_update(since_id)
        else:
            meta = store.latest()
        if meta is None:
            return JSONResponse({"item": None})
        return JSONResponse({"item": meta.public()})

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: int) -> Response:
        meta = store.get(frame_id)
        if meta is None:
            # The id is unknown — most likely a stale id from a
            # previous cc instance (FrameStore rebuilt with fresh
            # mtimes) or evicted from the ring buffer. Tell the
            # client what's actually available so it can resync.
            latest = store.latest()
            raise HTTPException(
                404,
                f"frame id {frame_id} not in store "
                f"(have {store.count()} frames; latest id="
                f"{latest.id if latest else 'none'})",
            )
        try:
            data = Path(meta.path).read_bytes()
        except FileNotFoundError:
            raise HTTPException(410, "frame file gone")
        return Response(content=data, media_type=_content_type_for(meta.path))

    @app.get("/api/frames/{frame_id}/neighbours")
    def frame_neighbours(frame_id: int) -> JSONResponse:
        prev_id, next_id = store.neighbours(frame_id)
        return JSONResponse({"prev": prev_id, "next": next_id})

    # ── runs ──────────────────────────────────────────────────────
    @app.post("/api/run")
    async def start_run(req: RunRequest) -> JSONResponse:
        try:
            record = await runner.start(
                intent=req.intent,
                no_focus=req.no_focus,
                vault=req.vault,
                password=req.password,
                vault_passphrase=req.vault_passphrase,
                skip_verify=req.skip_verify,
                platform=req.platform,
                dry_run=req.dry_run,
                allow_llm_fallback=req.allow_llm_fallback,
                planner=req.planner,
                ml_adapter=req.ml_adapter,
            )
        except RunnerBusy as e:
            raise HTTPException(409, str(e))
        return JSONResponse(record.public())

    @app.get("/api/runs")
    def list_runs(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse({
            "items": [r.public() for r in runner.list(limit=limit)],
        })

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        r = runner.get(run_id)
        if r is None:
            raise HTTPException(404, "run not found")
        return JSONResponse(r.public())

    @app.get("/api/runs/{run_id}/logs")
    async def run_logs(
        run_id: str, request: Request,
    ) -> StreamingResponse:
        if runner.get(run_id) is None:
            raise HTTPException(404, "run not found")

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for ev in bus.subscribe_run(run_id, replay=True):
                    if await request.is_disconnected():
                        break
                    yield _sse(None, ev.public())
                yield _sse("done", {"run_id": run_id})
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/logs")
    async def logs_global(
        request: Request, tail: int = Query(200, ge=0, le=2000),
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[bytes]:
            try:
                async for ev in bus.subscribe_global(replay_tail=tail):
                    if await request.is_disconnected():
                        break
                    yield _sse(None, ev.public())
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── manual mouse control ─────────────────────────────────────
    # Lets the UI drive the host cursor directly: click a point on the
    # screenshot or fire a button. Refuses while a run is in flight
    # because the runner owns the mouse for that window.
    #
    # Post-action snapshot policy:
    # The host might react instantly (a window focuses, a button
    # depresses) or seconds later (a page loads, a context menu
    # animates in, a modal renders, an app launches). We can't tell
    # which up front. Instead of a fixed sleep we run a
    # poll-until-stable loop: grab the first frame, then keep grabbing
    # at ``_POLL_INTERVAL_S`` intervals until two consecutive frames
    # are pixel-stable (normalised MSE < ``_STABLE_MSE_THR``) or the
    # ``_MAX_WAIT_S`` budget is exhausted. Each "interesting" frame
    # (initial + every changed frame + last stable frame) is written
    # so the UI replay shows the full sequence; near-duplicates of
    # the previous saved frame are skipped to keep the watch dir
    # readable.
    import os as _os
    _POLL_INTERVAL_S = max(
        0.25,
        float(_os.environ.get("HANDSNEYES_CC_POLL_INTERVAL_S", "1.5")),
    )
    _MAX_WAIT_S = max(
        _POLL_INTERVAL_S,
        float(_os.environ.get("HANDSNEYES_CC_MAX_WAIT_S", "15.0")),
    )
    # "How much of the image changed" — fraction of cells in a
    # downsampled 64×64 luminance grid whose absolute delta exceeds
    # ``_CELL_DELTA_THR`` (out of 255). Downsampling kills small
    # localised changes (cursor wiggling a few pixels, webcam shimmer)
    # because they collapse to <1 cell; real UI events (popup, menu,
    # page load, focus highlight) move whole regions so 1–50% of
    # cells flip. Default 0.5% catches popups while ignoring a moving
    # cursor.
    _CHANGE_FRACTION_THR = max(
        0.0,
        float(_os.environ.get("HANDSNEYES_CC_CHANGE_FRACTION", "0.005")),
    )
    # Dedup mode (used by /api/snapshot?dedup=1 and the typing /
    # active-refresh loops) needs to be more sensitive than the
    # post-mouse-action stability check: a single typed character in
    # a text field only flips ~3 cells of the 64×64 grid (~0.07%),
    # well below the post-action 0.5% noise gate. Default 0.001 (4
    # cells) catches that while still ignoring cursor-only motion.
    _DEDUP_FRACTION_THR = max(
        0.0,
        float(
            _os.environ.get("HANDSNEYES_CC_DEDUP_FRACTION", "0.001"),
        ),
    )
    _CELL_DELTA_THR = max(
        1,
        int(_os.environ.get("HANDSNEYES_CC_CELL_DELTA", "16")),
    )
    _DOWNSAMPLE = max(
        16,
        int(_os.environ.get("HANDSNEYES_CC_DOWNSAMPLE", "64")),
    )

    def _changed_fraction(a, b) -> float:
        """Fraction of cells in a downsampled grayscale grid that
        moved by more than ``_CELL_DELTA_THR``. Robust to small
        cursor displacements and webcam noise; sensitive to any
        region-scale UI change."""
        import cv2 as _cv2
        import numpy as np
        try:
            if a.shape != b.shape:
                return 1.0
            ga = _cv2.cvtColor(a, _cv2.COLOR_BGR2GRAY)
            gb = _cv2.cvtColor(b, _cv2.COLOR_BGR2GRAY)
            da = _cv2.resize(
                ga, (_DOWNSAMPLE, _DOWNSAMPLE),
                interpolation=_cv2.INTER_AREA,
            )
            db = _cv2.resize(
                gb, (_DOWNSAMPLE, _DOWNSAMPLE),
                interpolation=_cv2.INTER_AREA,
            )
            diff = np.abs(da.astype("int16") - db.astype("int16"))
            return float(np.mean(diff > _CELL_DELTA_THR))
        except Exception:
            return 1.0

    async def _snapshot_after_manual_action(label: str) -> None:
        """Save the initial frame, poll until the host screen settles
        (or the timeout elapses), then save the *final* frame only if
        it differs from the initial. Intermediary frames during the
        transition are not persisted — the UI just shows
        before-and-after.

        "Settled" = a downsampled-grid change-fraction check between
        the two most recent polls drops below ``_CHANGE_FRACTION_THR``
        for two consecutive polls. This is deliberately not pixel-MSE
        — a moving cursor or a few jittered webcam pixels won't trip
        it, but any region-scale UI change (popup, menu, page load,
        focused field highlight) will.

        Capture is serialized; the webcam stays open across the poll
        sleeps to avoid open/close cost and keep a consistent view.
        """
        if settings is None:
            return
        if runner.is_busy():
            # Capture would race with the run's own webcam handle.
            return
        async with _manual_capture_lock:
            from datetime import datetime
            import cv2
            # Honour the active target's capture_source. Without this,
            # manual snapshots always grabbed via the webcam — fooling
            # the operator with SMPTE bars whenever a screen-capture
            # target was configured.
            use_self = bool(
                app.state.runtime_state.get("use_self_capture", False)
            )
            resolution = None
            if (settings.capture.resolution_width
                    and settings.capture.resolution_height):
                resolution = (
                    settings.capture.resolution_width,
                    settings.capture.resolution_height,
                )
            if use_self:
                from handsneyes.core.capture.screen import ScreenCapture
                cap = ScreenCapture(
                    display_index=settings.capture.device_index,
                )
            else:
                from handsneyes.core.capture.webcam import WebcamCapture
                cap = WebcamCapture(
                    device_index=settings.capture.device_index,
                    resolution=resolution,
                )
            out_dir = store.watch_dir / "manual"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return

            def _write(image, suffix: str) -> bool:
                seq = int(datetime.now().timestamp() * 1000) % 10_000
                ts = datetime.now().strftime("%H%M%S")
                tag = label if not suffix else f"{label}_{suffix}"
                path = out_dir / f"{seq:04d}_{ts}_{tag}.png"
                try:
                    ok = cv2.imwrite(str(path), image)
                    if not ok:
                        logger.warning(
                            "imwrite returned False for %s", path,
                        )
                    return bool(ok)
                except Exception as e:
                    logger.warning("imwrite failed for %s: %s", path, e)
                    return False

            try:
                await cap.open()
                # Let the cursor settle visually before grabbing.
                await asyncio.sleep(0.25)
                try:
                    first = await cap.capture_frame()
                except Exception as e:
                    logger.warning("manual snapshot failed: %s", e)
                    return
                _write(first.image, "")
                prev = first.image
                cur_img = first.image
                t0 = asyncio.get_event_loop().time()
                stable_streak = 0
                while True:
                    elapsed = asyncio.get_event_loop().time() - t0
                    if elapsed >= _MAX_WAIT_S:
                        break
                    await asyncio.sleep(_POLL_INTERVAL_S)
                    try:
                        cur = await cap.capture_frame()
                    except Exception as e:
                        logger.debug("poll capture failed: %s", e)
                        continue
                    cur_img = cur.image
                    frac = _changed_fraction(prev, cur_img)
                    if frac < _CHANGE_FRACTION_THR:
                        stable_streak += 1
                    else:
                        stable_streak = 0
                    prev = cur_img
                    if stable_streak >= 2:
                        break
                # Save the final frame only if it actually differs
                # from the initial one — otherwise the action was a
                # no-op and the first frame already shows the state.
                elapsed = asyncio.get_event_loop().time() - t0
                if _changed_fraction(first.image, cur_img) \
                        >= _CHANGE_FRACTION_THR:
                    _write(cur_img, f"t{elapsed:.1f}s_final")
            finally:
                try:
                    await cap.close()
                except Exception:
                    pass

    async def _snapshot_single_dedup(label: str) -> bool:
        """Grab exactly one frame and persist it only if it differs
        from the most recent existing frame in the store. Used by
        the UI's "Active Refresh" idle poller — we don't need a
        poll-until-stable loop for an unsolicited capture, we just
        want to record state transitions and skip duplicates.

        Returns True iff a frame was written.
        """
        if settings is None or runner.is_busy():
            return False
        async with _manual_capture_lock:
            from datetime import datetime
            import cv2
            use_self = bool(
                app.state.runtime_state.get("use_self_capture", False)
            )
            resolution = None
            if (settings.capture.resolution_width
                    and settings.capture.resolution_height):
                resolution = (
                    settings.capture.resolution_width,
                    settings.capture.resolution_height,
                )
            if use_self:
                from handsneyes.core.capture.screen import ScreenCapture
                cap = ScreenCapture(
                    display_index=settings.capture.device_index,
                )
            else:
                from handsneyes.core.capture.webcam import WebcamCapture
                cap = WebcamCapture(
                    device_index=settings.capture.device_index,
                    resolution=resolution,
            )
            out_dir = store.watch_dir / "manual"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return False
            try:
                await cap.open()
                await asyncio.sleep(0.25)
                try:
                    frame = await cap.capture_frame()
                except Exception as e:
                    logger.warning("idle snapshot failed: %s", e)
                    return False
            finally:
                try:
                    await cap.close()
                except Exception:
                    pass

            # Compare against the most recent frame in the store —
            # regardless of which run/source produced it. If nothing
            # has changed, skip the write so the watch dir doesn't
            # fill with duplicates of an idle screen.
            latest = store.latest()
            if latest is not None:
                try:
                    prior = cv2.imread(str(latest.path))
                except Exception:
                    prior = None
                if prior is not None and \
                        _changed_fraction(prior, frame.image) \
                        < _DEDUP_FRACTION_THR:
                    return False

            seq = int(datetime.now().timestamp() * 1000) % 10_000
            ts = datetime.now().strftime("%H%M%S")
            path = out_dir / f"{seq:04d}_{ts}_{label}.png"
            try:
                ok = cv2.imwrite(str(path), frame.image)
                if not ok:
                    logger.warning(
                        "imwrite returned False for %s", path,
                    )
                return bool(ok)
            except Exception as e:
                logger.warning("imwrite failed for %s: %s", path, e)
                return False

    def _commander_cfg():
        if settings is None:
            class _Default:
                pi_base_url = "http://10.0.0.2:8080"
                transport = "bt"
                screen_width = 1920
                screen_height = 1080
            return _Default()
        return settings.commander

    async def _with_mouse(action):
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        from handsneyes.io.mouse.backends.http import HttpMouseOutput
        cfg = _commander_cfg()
        mouse = HttpMouseOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        async with _manual_mouse_lock:
            try:
                await mouse.connect()
            except Exception as e:
                raise HTTPException(502, f"mouse connect failed: {e}")
            try:
                return await action(mouse)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"mouse action failed: {e}")
            finally:
                try:
                    await mouse.disconnect()
                except Exception:
                    pass

    def _schedule_snapshot(label: str) -> None:
        asyncio.create_task(_snapshot_after_manual_action(label))

    @app.post("/api/mouse/drag")
    async def mouse_drag(req: MouseDragRequest) -> JSONResponse:
        """Drag-and-drop: home to (from), press button, home to (to),
        release. Same single-host-mouse exclusivity as click_at."""
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        from handsneyes.core.vision.session_adapter import SessionAdapter as _SessionAdapter
        from handsneyes.core.vision.visual_servo_homer import (
            VisualServoHomer,
        )

        async with _manual_mouse_lock:
            ctx = keyboard = mouse = capture = None
            try:
                ctx, keyboard, mouse, capture = await context_factory()
            except Exception as e:
                if capture is not None:
                    try: await capture.close()
                    except Exception: pass
                if keyboard is not None:
                    try: await keyboard.disconnect()
                    except Exception: pass
                if mouse is not None:
                    try: await mouse.disconnect()
                    except Exception: pass
                raise HTTPException(502, f"context_factory failed: {e}")
            try:
                adapter = _SessionAdapter(ctx)
                homer = VisualServoHomer(session=adapter)
                outcome = await homer.drag_to_pixels(
                    req.from_x_pct, req.from_y_pct,
                    req.to_x_pct, req.to_y_pct,
                    button=req.button,
                )
                import time as _time
                # A successful drag leaves the cursor near (to_x, to_y),
                # so update the no-slam cache to that point.
                if bool(outcome.clicked):
                    app.state.last_click_xy_at = (
                        (req.to_x_pct, req.to_y_pct), _time.time(),
                    )
                    app.state.last_scroll_home_xy = (
                        req.to_x_pct, req.to_y_pct,
                    )
                return JSONResponse({
                    "ok": bool(outcome.clicked),
                    "reason": outcome.reason,
                    "steps": outcome.steps,
                    "from": [req.from_x_pct, req.from_y_pct],
                    "to":   [req.to_x_pct, req.to_y_pct],
                    "button": req.button,
                })
            except Exception as e:
                logger.exception("drag_to_pixels failed")
                raise HTTPException(502, f"drag_to_pixels failed: {e}")
            finally:
                if capture is not None:
                    try: await capture.close()
                    except Exception: pass
                if keyboard is not None:
                    try: await keyboard.disconnect()
                    except Exception: pass
                if mouse is not None:
                    try: await mouse.disconnect()
                    except Exception: pass

    @app.post("/api/mouse/click_at")
    async def mouse_click_at(req: MouseClickAtRequest) -> JSONResponse:
        """Closed-loop visual-servo click at a webcam-image pixel.

        Routes through ``VisualServoHomer.home_to_pixel`` so the cursor
        is actually homed to the supplied pixel using the same CV that
        the controller uses. Open-loop ``MouseOutput.click_at`` was
        wrong on macOS because BT HID relative moves are subject to
        non-linear pointer acceleration.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        # Late import: pulls heavy CV deps only when actually used.
        from handsneyes.core.vision.session_adapter import SessionAdapter as _SessionAdapter
        from handsneyes.core.vision.visual_servo_homer import (
            VisualServoHomer,
        )

        async with _manual_mouse_lock:
            ctx = keyboard = mouse = capture = None
            try:
                ctx, keyboard, mouse, capture = await context_factory()
            except Exception as e:
                if capture is not None:
                    try: await capture.close()
                    except Exception: pass
                if keyboard is not None:
                    try: await keyboard.disconnect()
                    except Exception: pass
                if mouse is not None:
                    try: await mouse.disconnect()
                    except Exception: pass
                raise HTTPException(502, f"context_factory failed: {e}")

            try:
                adapter = _SessionAdapter(ctx)
                homer = VisualServoHomer(session=adapter)
                # No-slam follow-up: if a click landed recently, the
                # cursor is still at that pixel on the host. Reusing
                # it as the starting cursor lets the next click stay
                # within whatever UI element the previous click
                # opened (menu, dialog, dropdown). Without this
                # every click_at re-slams to the corner, which
                # dismisses transient UI like LibreOffice's menus.
                NO_SLAM_CACHE_TTL_S = 30.0
                # Same TTL for the homer's learned pct-per-HID ratio
                # — host-specific calibration that survives across
                # consecutive click_at calls.
                RATIO_CACHE_TTL_S = 300.0
                import time as _time
                prev_cursor_pct = None
                cached = app.state.last_click_xy_at
                if cached is not None:
                    cached_xy, cached_t = cached
                    if _time.time() - cached_t < NO_SLAM_CACHE_TTL_S:
                        prev_cursor_pct = cached_xy
                # Seed the homer's pct-per-HID ratios from the
                # previous click's learned values, if recent. Cuts
                # the closed-loop's ratio-discovery phase (~5–10
                # steps of overshoot + EMA refinement against the
                # DEFAULT seed) on every subsequent click within
                # the session.
                cached_ratio = app.state.last_homer_ratio_at
                if cached_ratio is not None:
                    payload, rt = cached_ratio
                    if _time.time() - rt < RATIO_CACHE_TTL_S:
                        # Backwards-compatible unpack — the payload
                        # has grown across versions:
                        #   payload[0:4] = closed-loop x,y + fast x,y
                        #   payload[4:6] = optional hotspot (x, y)
                        #   payload[6:8] = optional openloop x, y
                        rx, ry, fx, fy = payload[:4]
                        homer._pct_per_hid_x = rx
                        homer._pct_per_hid_y = ry
                        homer._pct_per_hid_fast_x = fx
                        homer._pct_per_hid_fast_y = fy
                        if len(payload) >= 6:
                            homer._calibrated_hotspot_offset = (
                                payload[4], payload[5],
                            )
                        if len(payload) >= 8:
                            # Openloop ratio is calibrated against
                            # the full-bulk send pattern (~1000 HID
                            # sustained chunked stream). It doesn't
                            # extrapolate cleanly from the servo's
                            # small-step ratio, so we cache it
                            # separately. Loading it here lets the
                            # 2nd-and-later openloop clicks skip
                            # the ~3 s calibration burst entirely.
                            homer._pct_per_hid_openloop_x = payload[6]
                            homer._pct_per_hid_openloop_y = payload[7]
                outcome = await homer.home_to_pixel(
                    req.x_pct, req.y_pct, button=req.button,
                    prev_cursor_pct=prev_cursor_pct,
                    click_count=req.count,
                )
                # click_at successfully landed the cursor at this
                # pixel — update the scroll-home cache so the next
                # /api/mouse/scroll at the same spot skips a fresh
                # home, AND the no-slam cache so the next click_at
                # within the TTL skips the slam phase entirely.
                if bool(outcome.clicked):
                    app.state.last_scroll_home_xy = (req.x_pct, req.y_pct)
                    app.state.last_click_xy_at = (
                        (req.x_pct, req.y_pct), _time.time(),
                    )
                    # Persist the homer's converged pct-per-HID
                    # ratios (both closed-loop AND fast-cruise) so
                    # the NEXT click starts already-calibrated
                    # instead of EMA-converging from DEFAULT all
                    # over again.
                    hsp = homer._calibrated_hotspot_offset
                    olx = getattr(homer, "_pct_per_hid_openloop_x", None)
                    oly = getattr(homer, "_pct_per_hid_openloop_y", None)
                    # Build payload: 4 mandatory ratios + optional
                    # hotspot pair + optional openloop pair. Hotspot
                    # is added unconditionally (with zero-tail if
                    # absent) when openloop is present, so the
                    # backwards-compat unpack can use len() checks.
                    payload = (
                        homer._pct_per_hid_x,
                        homer._pct_per_hid_y,
                        homer._pct_per_hid_fast_x,
                        homer._pct_per_hid_fast_y,
                    )
                    if hsp is not None or (olx is not None and oly is not None):
                        payload = payload + (
                            hsp[0] if hsp is not None else 0.0,
                            hsp[1] if hsp is not None else 0.0,
                        )
                    if olx is not None and oly is not None:
                        payload = payload + (olx, oly)
                    app.state.last_homer_ratio_at = (
                        payload, _time.time(),
                    )
                    # Every successful click produced a fresh
                    # history.jsonl trajectory — a new training row
                    # for the homer's retrain pipeline.
                    app.state.n_trajectories_since_train += 1
                    # Multi-click: the homer fired all clicks back-
                    # to-back inside its geometric-confirm block,
                    # BEFORE the 0.4 s proof sleep — so macOS sees
                    # them inside the double-click window (~250 ms).
                    # If we fired the extras here instead, the proof
                    # sleep would split the gap past the threshold
                    # and macOS would register independent single
                    # clicks instead of a double.
                # Drop a post-click frame at the watch-dir top level
                # so FrameStore (one-level-deep scan) picks it up and
                # the UI long-poll refreshes. We save THREE artefacts
                # per click:
                #   - <ts>_click_at.png        clean frame for training
                #   - <ts>_click_at_marked.png same frame with a cursor-
                #     icon overlay at the INTENDED click position, so
                #     a human can scan a directory of proofs and see
                #     at a glance whether the host cursor (in the
                #     frame) and the operator's target (overlay) line
                #     up
                #   - <ts>_click_at.json       sidecar with the pixel
                #     coordinates of the intended click + the homer's
                #     verified final cursor centroid; lets future ML
                #     work treat the proof directory as a labelled
                #     dataset without re-parsing image overlays
                try:
                    from datetime import datetime
                    import cv2
                    import json
                    await asyncio.sleep(0.35)
                    frame = await capture.capture_frame()
                    out_dir = store.watch_dir / "manual"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    seq = int(datetime.now().timestamp() * 1000) % 10_000
                    ts = datetime.now().strftime("%H%M%S")
                    stem = f"{seq:04d}_{ts}_click_at"
                    clean_path = out_dir / f"{stem}.png"
                    marked_path = out_dir / f"{stem}_marked.png"
                    json_path = out_dir / f"{stem}.json"
                    # Clean frame first — preserve for downstream ML.
                    cv2.imwrite(str(clean_path), frame.image)

                    img = frame.image
                    h_img, w_img = img.shape[:2]
                    aim_px_x = int(req.x_pct * w_img)
                    aim_px_y = int(req.y_pct * h_img)
                    final = outcome.final_cursor_pct
                    final_px = (
                        (int(final[0] * w_img), int(final[1] * h_img))
                        if final is not None else None
                    )

                    # Marker: bright cyan ring + crosshair at the
                    # operator's INTENDED pixel. Cyan is rare in
                    # most UI palettes so it pops against typical
                    # backgrounds without being confusable with the
                    # red HSV cursor accent.
                    overlay = img.copy()
                    CYAN = (255, 255, 0)
                    cv2.circle(overlay, (aim_px_x, aim_px_y), 14, CYAN, 2)
                    cv2.line(overlay, (aim_px_x - 20, aim_px_y),
                             (aim_px_x + 20, aim_px_y), CYAN, 1)
                    cv2.line(overlay, (aim_px_x, aim_px_y - 20),
                             (aim_px_x, aim_px_y + 20), CYAN, 1)
                    cv2.putText(
                        overlay, "click", (aim_px_x + 18, aim_px_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA,
                    )
                    # Homer's verified cursor centroid in green —
                    # if this disagrees with the click marker, the
                    # homer's residual was nonzero at commit.
                    if final_px is not None:
                        GREEN = (0, 255, 0)
                        cv2.circle(overlay, final_px, 8, GREEN, 2)
                        cv2.putText(
                            overlay, "homer", (final_px[0] + 12, final_px[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA,
                        )
                    cv2.imwrite(str(marked_path), overlay)

                    # Sidecar JSON for training.
                    sidecar = {
                        "ts": ts,
                        "frame_size": [w_img, h_img],
                        "intended_click_pct": [req.x_pct, req.y_pct],
                        "intended_click_px": [aim_px_x, aim_px_y],
                        "homer_final_cursor_pct": (
                            list(final) if final is not None else None
                        ),
                        "homer_final_cursor_px": (
                            list(final_px) if final_px is not None else None
                        ),
                        "outcome": outcome.reason,
                        "steps": outcome.steps,
                        "button": req.button,
                        "count": req.count,
                    }
                    with open(json_path, "w") as f:
                        json.dump(sidecar, f, indent=2)
                except Exception as e:
                    logger.warning("post-click snapshot failed: %s", e)
                return JSONResponse({
                    "ok": bool(outcome.clicked),
                    "reason": outcome.reason,
                    "steps": outcome.steps,
                    "x_pct": req.x_pct, "y_pct": req.y_pct,
                    "button": req.button,
                    "count": req.count,
                })
            except Exception as e:
                logger.exception("home_to_pixel failed")
                raise HTTPException(502, f"home_to_pixel failed: {e}")
            finally:
                if capture is not None:
                    try: await capture.close()
                    except Exception: pass
                if keyboard is not None:
                    try: await keyboard.disconnect()
                    except Exception: pass
                if mouse is not None:
                    try: await mouse.disconnect()
                    except Exception: pass

    @app.post("/api/mouse/click")
    async def mouse_click(req: MouseClickRequest) -> JSONResponse:
        # Inter-click gap for multi-click. macOS double-click threshold
        # is ~500 ms but most apps treat anything under ~250 ms as a
        # double. 80 ms keeps us safely inside that AND leaves enough
        # time for the Pi's BT HID report + ack so the two clicks
        # land as a distinct down-up-down-up pattern, not a coalesced
        # long press.
        gap = 0.08

        async def go(mouse):
            await mouse.click(req.button)
            for _ in range(1, req.count):
                await asyncio.sleep(gap)
                await mouse.click(req.button)
            return JSONResponse({
                "ok": True, "button": req.button, "count": req.count,
            })

        try:
            return await _with_mouse(go)
        finally:
            tag = (
                f"manual_click_{req.button}"
                if req.count == 1
                else f"manual_click_{req.button}_x{req.count}"
            )
            _schedule_snapshot(tag)

    @app.post("/api/mouse/move")
    async def mouse_move(req: MouseMoveRequest) -> JSONResponse:
        async def go(mouse):
            await mouse.move(req.dx, req.dy)
            return JSONResponse({"ok": True, "dx": req.dx, "dy": req.dy})

        try:
            return await _with_mouse(go)
        finally:
            # Manual mouse-move invalidates the no-slam click cache —
            # the cursor is no longer at the previously-clicked pixel.
            app.state.last_click_xy_at = None
            _schedule_snapshot("manual_move")

    # Tolerance (in normalised coords, both axes) for treating two
    # hover positions as "the same target" so we don't re-home on
    # every wheel event in a continuous gesture. 5 % ≈ 96 px on a
    # 1920-wide screen — well within a scrollable pane.
    SCROLL_HOME_TOL = 0.05

    @app.post("/api/mouse/scroll")
    async def mouse_scroll(req: MouseScrollRequest) -> JSONResponse:
        """Forward a wheel-tick to the target.

        When ``x_pct`` / ``y_pct`` are provided AND differ from the
        last successful home position by more than
        ``SCROLL_HOME_TOL`` in either axis, the cursor is first
        visually homed (no click) to the hover position so the
        scroll lands on the content the operator pointed at — not
        on whatever scrollable region the cursor was last left in.

        Subsequent scrolls within tolerance reuse the cached home
        and skip straight to ``mouse.scroll(amount)`` so a
        continuous gesture pays the homing cost exactly once.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        pos_specified = (
            req.x_pct is not None and req.y_pct is not None
        )
        last = getattr(app.state, "last_scroll_home_xy", None)
        needs_home = pos_specified and (
            last is None
            or abs(last[0] - req.x_pct) > SCROLL_HOME_TOL
            or abs(last[1] - req.y_pct) > SCROLL_HOME_TOL
        )

        if not needs_home:
            # Fast path: just send the wheel ticks, no webcam, no
            # homer. Fan amount out into |amount| single-tick reports
            # with a short sleep between them — matches the working
            # ScrollAgent pattern (agents/scroll.py) and produces the
            # macOS-acceleration "this is a gesture" effect.
            # Sending one big scroll(N) report was being interpreted
            # by macOS as a single notch with magnitude N, which it
            # caps to a tiny visual scroll.
            sign = 1 if req.amount > 0 else -1
            ticks = abs(req.amount)

            async def go(mouse):
                for _ in range(ticks):
                    await mouse.scroll(sign)
                    await asyncio.sleep(0.05)
                return JSONResponse({
                    "ok": True, "amount": req.amount,
                    "ticks_sent": ticks,
                    "x_pct": req.x_pct, "y_pct": req.y_pct,
                    "homed": False,
                })

            tag = "manual_scroll"
            if pos_specified:
                tag = (
                    f"manual_scroll_{int(req.x_pct * 100):02d}_"
                    f"{int(req.y_pct * 100):02d}"
                )
            try:
                resp = await _with_mouse(go)
            finally:
                # Synchronous post-action snapshot so the response
                # returns *after* a fresh frame is on disk. Without
                # this, the cc UI showed stale screenshots until
                # FrameStore polled (~250 ms later) — visually it
                # looked like "scroll did nothing." Cost is one
                # webcam open + grab (~500 ms) per scroll, which is
                # already coalesced from many wheel events.
                await _snapshot_after_manual_action(tag)
            return resp

        # Slow path: home cursor to (x_pct, y_pct) THEN scroll.
        # Same fixture as click_at — context_factory builds a full
        # ctx, VisualServoHomer.home_to_pixel(click=False) lands
        # the cursor without firing a button, then mouse.scroll(...).
        from handsneyes.core.vision.session_adapter import SessionAdapter as _SessionAdapter
        from handsneyes.core.vision.visual_servo_homer import (
            VisualServoHomer,
        )

        async with _manual_mouse_lock:
            ctx = keyboard = mouse = capture = None
            try:
                ctx, keyboard, mouse, capture = await context_factory()
            except Exception as e:
                if capture is not None:
                    try: await capture.close()
                    except Exception: pass
                if keyboard is not None:
                    try: await keyboard.disconnect()
                    except Exception: pass
                if mouse is not None:
                    try: await mouse.disconnect()
                    except Exception: pass
                raise HTTPException(502, f"context_factory failed: {e}")
            try:
                adapter = _SessionAdapter(ctx)
                homer = VisualServoHomer(session=adapter)
                outcome = await homer.home_to_pixel(
                    req.x_pct, req.y_pct, click=False,
                )
                homed_ok = bool(getattr(outcome, "clicked", False)) or (
                    # home_to_pixel's ClickOutcome semantically tracks
                    # "did we successfully land on the pixel" via
                    # the clicked flag even when click=False — the
                    # homer only sets it after the geometric confirm
                    # passes. Treat it as the home-success signal.
                    False
                )
                if homed_ok:
                    app.state.last_scroll_home_xy = (
                        req.x_pct, req.y_pct,
                    )
                # Whether the home succeeded or not, fire the scroll
                # — the operator clearly wants something to scroll
                # and a partial home is usually still in the right
                # general region. Fan out as single-tick reports
                # (see fast path for rationale).
                sign = 1 if req.amount > 0 else -1
                ticks = abs(req.amount)
                for _ in range(ticks):
                    await mouse.scroll(sign)
                    await asyncio.sleep(0.05)
                return JSONResponse({
                    "ok": True, "amount": req.amount,
                    "ticks_sent": ticks,
                    "x_pct": req.x_pct, "y_pct": req.y_pct,
                    "homed": True,
                    "home_ok": homed_ok,
                    "home_reason": getattr(outcome, "reason", ""),
                })
            except Exception as e:
                logger.exception("home-then-scroll failed")
                raise HTTPException(502, f"scroll failed: {e}")
            finally:
                # Best-effort post-action snapshot via the same lock
                # so it's serialised against the next mouse action.
                try:
                    from datetime import datetime
                    import cv2
                    await asyncio.sleep(0.3)
                    frame = await capture.capture_frame()
                    out_dir = store.watch_dir / "manual"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    seq = int(datetime.now().timestamp() * 1000) % 10_000
                    ts = datetime.now().strftime("%H%M%S")
                    path = out_dir / f"{seq:04d}_{ts}_scroll.png"
                    cv2.imwrite(str(path), frame.image)
                except Exception as e:
                    logger.warning("post-scroll snapshot failed: %s", e)
                if capture is not None:
                    try: await capture.close()
                    except Exception: pass
                if keyboard is not None:
                    try: await keyboard.disconnect()
                    except Exception: pass
                if mouse is not None:
                    try: await mouse.disconnect()
                    except Exception: pass

    # ── manual keyboard control ──────────────────────────────────
    async def _with_keyboard(action):
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        from handsneyes.io.keyboard.backends.http import HttpKeyboardOutput
        cfg = _commander_cfg()
        kb = HttpKeyboardOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        try:
            await kb.connect()
        except Exception as e:
            raise HTTPException(502, f"keyboard connect failed: {e}")
        try:
            return await action(kb)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"keyboard action failed: {e}")
        finally:
            try:
                await kb.disconnect()
            except Exception:
                pass

    @app.post("/api/keyboard/text")
    async def keyboard_text(req: KeyboardTextRequest) -> JSONResponse:
        async def go(kb):
            try:
                await kb.send_text(
                    req.text, warmup=req.warmup, secret=req.secret,
                )
            except TypeError:
                # Older backend signature without the kwargs.
                await kb.send_text(req.text, warmup=req.warmup)
            if req.append_enter:
                await kb.send_keystroke("Enter")
            return JSONResponse({
                "ok": True, "length": len(req.text),
                "secret": req.secret, "append_enter": req.append_enter,
            })
        return await _with_keyboard(go)

    @app.post("/api/keyboard/key")
    async def keyboard_key(req: KeyboardKeyRequest) -> JSONResponse:
        async def go(kb):
            if req.modifiers:
                await kb.send_key_combo(req.modifiers, req.key)
            else:
                await kb.send_keystroke(req.key)
            return JSONResponse({
                "ok": True, "key": req.key, "modifiers": req.modifiers,
            })
        return await _with_keyboard(go)

    # ── paste-file: type a local file's contents on the host ────
    @app.post("/api/paste-file")
    async def paste_file(req: PasteFileRequest) -> JSONResponse:
        """Type a local file's contents into a focused terminal on the
        host, then verify the round-trip via OCR.

        Sequence:
          1. Maximize the focused window (optional).
          2. ``base64 -d > {path}`` + Enter — start a decoder reading
             stdin.
          3. Type the base64-encoded content in 76-col lines. Base64
             is restricted to ``[A-Za-z0-9+/=]``, all of which have
             HID scancodes — so any byte content (including Unicode
             box-drawing or other chars that have no key mapping)
             survives the wire.
          4. Ctrl+D — close stdin, base64 decodes the buffer and
             writes the original bytes to the file.
          5. ``shasum -a 256 {path}`` + Enter — print framed SHA.
          6. Capture webcam, OCR the framed hash, compare. On
             mismatch, drive the chunked-MD5 repair loop.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        if settings is None:
            raise HTTPException(500, "settings not wired into app")

        from handsneyes.core.capture.webcam import WebcamCapture
        from handsneyes.io.keyboard.backends.http import HttpKeyboardOutput

        cfg = _commander_cfg()
        # Generous per-request timeout: each /bt/text call types
        # the whole payload character-by-character via BT HID. A
        # 76-col base64 line takes a few seconds with the warmup
        # pre-flight; larger payloads (chunk overwrites) need more
        # headroom than the default 10 s.
        kb = HttpKeyboardOutput(
            base_url=cfg.pi_base_url,
            timeout=120.0,
            transport=cfg.transport,
        )
        capture = None
        if req.verify:
            resolution = None
            if (settings.capture.resolution_width
                    and settings.capture.resolution_height):
                resolution = (
                    settings.capture.resolution_width,
                    settings.capture.resolution_height,
                )
            capture = WebcamCapture(
                device_index=settings.capture.device_index,
                resolution=resolution,
            )

        async def _cleanup():
            if capture is not None:
                try:
                    await capture.close()
                except Exception:
                    pass
            try:
                await kb.disconnect()
            except Exception:
                pass

        try:
            await kb.connect()
            if capture is not None:
                await capture.open()
        except Exception as e:
            await _cleanup()
            raise HTTPException(502, f"paste-file connect failed: {e}")

        try:
            # 1) Maximize the focused window.
            if req.maximize:
                if req.platform == "macos":
                    # Cmd+Ctrl+F — native macOS full-screen toggle.
                    await kb.send_key_combo(["ctrl", "meta"], "f")
                else:
                    # GNOME: Super+Up maximises.
                    await kb.send_key_combo(["meta"], "Up")
                await asyncio.sleep(1.0)

            # 2-3) Send the body as base64 piped through ``base64 -d``
            # on the host. Typing raw content character-by-character
            # would fail on any byte without an HID scancode (Unicode
            # box-drawing, em-dash, etc.) — and would also break on
            # operator-side shell-special chars the moment the host
            # echoes them. Base64's charset is ``[A-Za-z0-9+/=]``,
            # entirely covered by the HID map, so any byte goes
            # through cleanly. The host decodes back to original
            # bytes — SHA over the original is unchanged.
            import base64 as _b64
            content_b64 = _b64.b64encode(req.content.encode("utf-8")).decode("ascii")
            # Standard MIME-style line wrap at 76 columns so each
            # line is well under the Pi's text-buffer ceiling and so
            # the terminal echo doesn't smear across the whole
            # screen. ``base64 -d`` ignores newlines transparently.
            B64_WRAP = 76
            b64_lines = [
                content_b64[i : i + B64_WRAP]
                for i in range(0, len(content_b64), B64_WRAP)
            ] or [""]

            await kb.send_text(f"base64 -d > {req.path}")
            await asyncio.sleep(0.05)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.35)

            for line in b64_lines:
                if line:
                    await kb.send_text(line)
                await kb.send_keystroke("Enter")
                await asyncio.sleep(0.03)

            # 4) Ctrl+D closes base64's stdin → it decodes the buffer
            # and writes the original bytes to ``req.path``.
            await kb.send_key_combo(["ctrl"], "d")
            await asyncio.sleep(0.35)

            result: dict = {
                "ok": True,
                "wrote_path": req.path,
                "sent_chars": len(req.content),
                "sent_lines": req.content.count("\n") + (
                    0 if req.content.endswith("\n") else 1
                ),
            }

            # 5) Verify + auto-repair via SHA-256 + chunked MD5 diff.
            # Replaces the old "cat file → SequenceMatcher" heuristic
            # with a cryptographic check whose verdict is
            # deterministic (modulo SHA collisions). Repair is
            # automatic — bad chunks are identified and overwritten
            # in place via base64 + dd seek=. Up to 3 rounds.
            if req.verify and capture is not None:
                from handsneyes.ui import paste_protocol as pp
                content_bytes = req.content.encode("utf-8")
                local_sha = pp.file_sha256(content_bytes)
                local_chunks = pp.chunk_hashes(content_bytes)
                nchunks = pp.n_chunks(len(content_bytes))
                # Persistence policy: keep retransmitting until SHA
                # converges. Bounded only by two safety guards:
                #   * per-chunk attempt cap — if one specific block
                #     refuses to land after this many retransmits in
                #     a row, the channel is broken for it and we
                #     give up rather than spinning forever.
                #   * no-progress detection — if the bad-chunk set
                #     is identical to the previous round AFTER we
                #     already overwrote those chunks, our writes
                #     aren't taking effect and we'd spin pointlessly.
                # The global round cap is generous so a busy channel
                # has room to converge.
                MAX_REPAIR_ROUNDS = 30
                PER_CHUNK_RETRY_CAP = 6

                async def _ocr_now(label: str) -> tuple[str, "Path"]:
                    """Capture, save under manual/, OCR, return (text, path)."""
                    from datetime import datetime
                    import cv2
                    frame = await capture.capture_frame()
                    out_dir = store.watch_dir / "manual"
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                    seq = int(datetime.now().timestamp() * 1000) % 10_000
                    ts = datetime.now().strftime("%H%M%S")
                    fpath = out_dir / f"{seq:04d}_{ts}_{label}.png"
                    try:
                        cv2.imwrite(str(fpath), frame.image)
                    except Exception as e:
                        logger.warning("imwrite failed: %s", e)
                    text = ""
                    try:
                        import pytesseract
                        text = pytesseract.image_to_string(frame.image)
                    except Exception as e:
                        logger.warning("OCR failed: %s", e)
                    return text, fpath

                async def _read_host_sha() -> tuple[str | None, "Path"]:
                    """Type the SHA-print command; OCR-retry up to 3
                    times since the hash line is small + structured
                    and an OCR miss is recoverable by reprinting."""
                    last_path = None
                    for ocr_try in range(3):
                        await kb.send_text(pp.cmd_sha_print(req.path))
                        await kb.send_keystroke("Enter")
                        await asyncio.sleep(1.4)
                        text, last_path = await _ocr_now(
                            f"paste_sha_r{ocr_try}",
                        )
                        h = pp.parse_sha_from_ocr(text)
                        if h is not None:
                            return h, last_path
                        logger.info(
                            "SHA OCR retry %d/3 — no parse", ocr_try + 1,
                        )
                    return None, last_path

                async def _read_host_chunks() -> tuple[dict[int, str], "Path"]:
                    last_path = None
                    for ocr_try in range(3):
                        await kb.send_text(
                            pp.cmd_chunks_print(req.path, nchunks),
                        )
                        await kb.send_keystroke("Enter")
                        # Give the loop time — each iteration does a
                        # dd + openssl. Crude estimate: ~80 ms per
                        # chunk, plus a startup tax.
                        await asyncio.sleep(0.6 + 0.08 * nchunks)
                        text, last_path = await _ocr_now(
                            f"paste_chunks_r{ocr_try}",
                        )
                        hashes = pp.parse_chunks_from_ocr(text)
                        if hashes:
                            return hashes, last_path
                        logger.info(
                            "chunks OCR retry %d/3 — no parse",
                            ocr_try + 1,
                        )
                    return {}, last_path

                rounds_log: list[dict] = []
                final_frame: Path | None = None
                matched = False

                def _emit(msg: str, level: str = "INFO") -> None:
                    """Publish progress to the LogBus so the SSE
                    stream shows it in the operator's log pane in
                    real time, not just when the endpoint returns."""
                    try:
                        from handsneyes.ui.log_bus import (
                            LogEvent,
                        )
                        import time as _time
                        bus.publish(LogEvent(
                            ts=_time.time(), level=level,
                            source="paste-file", msg=msg, run_id=None,
                        ))
                    except Exception:
                        pass

                _emit(
                    f"verify start: {nchunks} chunks @ "
                    f"{pp.CHUNK_SIZE}B, local SHA={local_sha[:12]}…",
                )

                # Per-chunk retransmit counter: index → times we've
                # rewritten this chunk. We escalate to "unrecoverable"
                # if any single chunk exceeds the cap.
                chunk_retry_count: dict[int, int] = {}
                last_bad_set: frozenset[int] | None = None

                for repair_round in range(MAX_REPAIR_ROUNDS + 1):
                    _emit(f"round {repair_round}: reading host SHA…")
                    host_sha, sha_frame = await _read_host_sha()
                    if sha_frame is not None:
                        final_frame = sha_frame
                    round_info: dict = {
                        "round": repair_round,
                        "host_sha": host_sha,
                        "local_sha": local_sha,
                    }
                    if host_sha == local_sha:
                        _emit(
                            f"round {repair_round}: ✓ SHA match "
                            f"({host_sha[:12]}…)",
                        )
                        round_info["match"] = True
                        rounds_log.append(round_info)
                        matched = True
                        break
                    round_info["match"] = False
                    rounds_log.append(round_info)
                    _emit(
                        f"round {repair_round}: SHA mismatch "
                        f"(host={(host_sha or '?')[:12]}…)",
                        level="WARNING",
                    )
                    if repair_round >= MAX_REPAIR_ROUNDS:
                        break

                    # Mismatch — identify bad chunks and overwrite.
                    _emit(
                        f"round {repair_round}: reading chunk hashes…",
                    )
                    host_chunks, chunks_frame = await _read_host_chunks()
                    if chunks_frame is not None:
                        final_frame = chunks_frame
                    diff = pp.diff_chunks(local_chunks, host_chunks)
                    # Defensive: chunks the OCR couldn't read at all
                    # are treated as bad too — they may be wrong.
                    bad = sorted(set(diff.bad_indices + diff.unknown_indices))
                    round_info["bad_indices"] = bad
                    round_info["unknown_indices"] = diff.unknown_indices
                    if not bad:
                        # SHA disagreed but per-chunk hashes all
                        # agree — usually OCR couldn't parse the
                        # chunk block at all. Bail rather than spin.
                        round_info["abort_reason"] = (
                            "no parseable bad chunks; "
                            "OCR likely failed on chunks block"
                        )
                        _emit(
                            f"round {repair_round}: aborting — "
                            f"chunk-hash OCR yielded nothing parseable",
                            level="ERROR",
                        )
                        break

                    # No-progress guard: if the bad set is identical
                    # to the prior round and we already rewrote those
                    # chunks, our writes aren't taking effect. Spinning
                    # further can only burn time.
                    current_bad = frozenset(bad)
                    if (last_bad_set is not None
                            and current_bad == last_bad_set
                            and repair_round >= 2):
                        round_info["abort_reason"] = (
                            f"no progress — same {len(bad)} chunks "
                            f"bad after retransmit"
                        )
                        _emit(
                            f"round {repair_round}: aborting — same "
                            f"{len(bad)} chunks bad as last round; "
                            f"channel not accepting writes",
                            level="ERROR",
                        )
                        break
                    last_bad_set = current_bad

                    # Per-chunk retry cap.
                    blocked = [
                        i for i in bad
                        if chunk_retry_count.get(i, 0) >= PER_CHUNK_RETRY_CAP
                    ]
                    if blocked:
                        round_info["abort_reason"] = (
                            f"chunks {blocked[:10]} exceeded "
                            f"{PER_CHUNK_RETRY_CAP} retransmit attempts"
                        )
                        round_info["unrecoverable_chunks"] = blocked
                        _emit(
                            f"round {repair_round}: aborting — "
                            f"{len(blocked)} chunk(s) refusing to land "
                            f"after {PER_CHUNK_RETRY_CAP} retransmits "
                            f"({blocked[:10]}…)",
                            level="ERROR",
                        )
                        break

                    _emit(
                        f"round {repair_round}: repairing "
                        f"{len(bad)}/{nchunks} chunks "
                        f"({len(diff.unknown_indices)} unknown)",
                    )

                    async def _overwrite_chunk(idx: int, payload: bytes):
                        """Stage payload via line-wrapped base64 →
                        write into place via ``dd seek=``. Splitting
                        the b64 across many small ``send_text`` calls
                        keeps each /bt/text request bounded; sending
                        the whole 2.7 KB inline (as the original
                        single-string command did) overran the HTTP
                        timeout on real BT HID."""
                        import base64 as _b64
                        tmp = "/tmp/_cc_overwrite.bin"
                        b64 = _b64.b64encode(payload).decode("ascii")
                        WRAP = 76
                        lines = [
                            b64[i : i + WRAP]
                            for i in range(0, len(b64), WRAP)
                        ] or [""]
                        await kb.send_text(f"base64 -d > {tmp}")
                        await kb.send_keystroke("Enter")
                        await asyncio.sleep(0.1)
                        for ln in lines:
                            if ln:
                                await kb.send_text(ln)
                            await kb.send_keystroke("Enter")
                            await asyncio.sleep(0.03)
                        await kb.send_key_combo(["ctrl"], "d")
                        await asyncio.sleep(0.15)
                        await kb.send_text(
                            f"dd if={tmp} of={req.path} "
                            f"bs={pp.CHUNK_SIZE} seek={idx} "
                            f"conv=notrunc 2>/dev/null && rm -f {tmp}"
                        )
                        await kb.send_keystroke("Enter")

                    for idx in bad:
                        start = idx * pp.CHUNK_SIZE
                        payload = content_bytes[start : start + pp.CHUNK_SIZE]
                        if not payload:
                            continue
                        await _overwrite_chunk(idx, payload)
                        chunk_retry_count[idx] = (
                            chunk_retry_count.get(idx, 0) + 1
                        )
                        # Small settle — the dd is bounded by chunk
                        # size, and the host shouldn't queue these.
                        await asyncio.sleep(0.25)

                # Sparse map — only chunks we actually retransmitted.
                retry_map = {
                    str(k): v for k, v in chunk_retry_count.items() if v > 0
                }
                result["verify"] = {
                    "match": matched,
                    "local_sha": local_sha,
                    "rounds": rounds_log,
                    "n_chunks": nchunks,
                    "chunk_size": pp.CHUNK_SIZE,
                    "max_repair_rounds": MAX_REPAIR_ROUNDS,
                    "per_chunk_retry_cap": PER_CHUNK_RETRY_CAP,
                    "chunk_retransmits": retry_map,
                    "frame": (final_frame.name if final_frame else None),
                }

            # 6) Optional pager-driven body readback. Independent of
            # (and additive to) the SHA verdict — for the operator
            # who wants to *see* the file scroll past on the webcam
            # rather than trust a hash. Bounded page count derived
            # from local line count.
            if req.body_readback and capture is not None:
                import difflib
                # Conservative: 30 visible lines per page after the
                # maximised terminal accounts for chrome + the
                # ``--More--`` prompt. Plus two extra pages for end-
                # of-file slop. Capped to keep runtime bounded.
                local_lines = req.content.count("\n") + 1
                pages_budget = max(2, (local_lines // 30) + 2)
                pages_budget = min(pages_budget, 60)
                _emit(
                    f"body readback: more {req.path} "
                    f"(~{pages_budget} pages)",
                )

                # Start fresh: clear the screen so the first page
                # OCR isn't polluted by SHA/CHUNKS framing tokens
                # left over from the verify section.
                await kb.send_text("clear")
                await kb.send_keystroke("Enter")
                await asyncio.sleep(0.35)
                await kb.send_text(f"more {req.path}")
                await kb.send_keystroke("Enter")
                await asyncio.sleep(0.9)

                pages_ocr: list[str] = []
                from datetime import datetime
                import cv2
                for p_idx in range(pages_budget):
                    try:
                        frame = await capture.capture_frame()
                    except Exception as e:
                        logger.warning("readback capture failed: %s", e)
                        break
                    out_dir = store.watch_dir / "manual"
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                    seq = int(datetime.now().timestamp() * 1000) % 10_000
                    ts = datetime.now().strftime("%H%M%S")
                    fpath = out_dir / f"{seq:04d}_{ts}_more_p{p_idx}.png"
                    try:
                        cv2.imwrite(str(fpath), frame.image)
                    except Exception as e:
                        logger.warning("imwrite failed: %s", e)
                    page_text = ""
                    try:
                        import pytesseract
                        page_text = pytesseract.image_to_string(frame.image)
                    except Exception as e:
                        logger.warning("readback OCR failed: %s", e)
                    pages_ocr.append(page_text)
                    # Advance to next page — Space, NOT Enter (Enter
                    # scrolls one line on more; Space goes a whole
                    # page).
                    await kb.send_text(" ")
                    await asyncio.sleep(0.55)

                # Defensive q in case the page budget was a hair too
                # generous and ``more`` is still alive at the
                # prompt. No-op if it's already exited.
                await kb.send_text("q")
                await asyncio.sleep(0.2)

                # Normalise & compare. OCR is lossy on body text
                # (unlike the SHA line's restricted charset), so the
                # similarity is an approximate sanity check, not a
                # cryptographic verdict. We:
                #   - drop empty lines (OCR routinely fabricates them)
                #   - rstrip each line (trailing whitespace is meaningless)
                #   - strip the pager's own ``--More--(NN%)`` prompt
                #     line so the readback isn't penalised for it
                import re as _re
                _MORE_PROMPT = _re.compile(
                    r"^\s*-{1,3}\s*More\s*-{1,3}\s*\(\s*\d+\s*%\s*\)\s*$",
                    _re.IGNORECASE,
                )

                def _norm_body(s: str) -> str:
                    out_lines = []
                    for ln in s.replace("\r", "").split("\n"):
                        ln = ln.rstrip()
                        if not ln.strip():
                            continue
                        if _MORE_PROMPT.match(ln):
                            continue
                        out_lines.append(ln)
                    return "\n".join(out_lines).strip()

                accumulated = "\n".join(pages_ocr)
                expected_norm = _norm_body(req.content)
                ocr_norm = _norm_body(accumulated)
                ratio = difflib.SequenceMatcher(
                    None, expected_norm, ocr_norm,
                ).ratio() if expected_norm else 0.0
                _emit(
                    f"body readback: similarity={ratio:.3f} "
                    f"(expected={len(expected_norm)}c, "
                    f"ocr={len(ocr_norm)}c, pages={len(pages_ocr)})",
                )
                result["body_readback"] = {
                    "pages": len(pages_ocr),
                    "similarity": round(ratio, 3),
                    "expected_chars": len(expected_norm),
                    "ocr_chars": len(ocr_norm),
                    # Bounded sample for UI.
                    "ocr_sample": accumulated[:2000],
                    # Full reconstructed-after-normalize text so an
                    # operator (or test) can run a real diff against
                    # the source — useful when SHA matches and they
                    # still want to *see* the byte-level recovery.
                    "recovered_text": ocr_norm,
                }

            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("paste-file failed")
            raise HTTPException(502, f"paste-file failed: {e}")
        finally:
            await _cleanup()

    @app.post("/api/snapshot")
    async def manual_snapshot(dedup: bool = False) -> JSONResponse:
        """Capture a fresh webcam frame on demand (no mouse action).

        ``?dedup=1`` switches from the poll-until-stable loop to a
        single-shot capture that's only persisted when it differs
        from the most recent frame in the store — used by the UI's
        Active Refresh idle poller to avoid filling the watch dir
        with duplicates of an unchanged screen.
        """
        if dedup:
            wrote = await _snapshot_single_dedup("idle_refresh")
            return JSONResponse({"ok": True, "wrote": wrote})
        await _snapshot_after_manual_action("manual_snapshot")
        return JSONResponse({"ok": True, "wrote": True})

    @app.post("/api/sync-text-from-host")
    async def sync_text_from_host(req: SyncTextRequest) -> JSONResponse:
        """OCR the focused host text field, return its current content.

        Used by the cc UI's passthrough mirror to recover from cursor-
        only edits (arrow keys, mid-string Backspace) that don't move
        enough pixels for the snapshot poll to fire. Defaults the
        region of interest to the last click_at position so just
        clicking into a text field is enough setup.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        if settings is None:
            raise HTTPException(503, "no settings configured")

        # Default ROI to last-clicked position (typical workflow:
        # click into a field, arrow around, hit sync).
        x_pct, y_pct = req.x_pct, req.y_pct
        if y_pct is None:
            cached = app.state.last_click_xy_at
            if cached is not None:
                (lx, ly), _ = cached
                if x_pct is None:
                    x_pct = lx
                y_pct = ly

        # One-shot capture — no poll-until-stable. Operator wants
        # immediate text feedback.
        async with _manual_capture_lock:
            from handsneyes.core.capture.webcam import WebcamCapture
            resolution = None
            if (settings.capture.resolution_width
                    and settings.capture.resolution_height):
                resolution = (
                    settings.capture.resolution_width,
                    settings.capture.resolution_height,
                )
            cap = WebcamCapture(
                device_index=settings.capture.device_index,
                resolution=resolution,
            )
            try:
                await asyncio.wait_for(cap.open(), timeout=10.0)
                await asyncio.sleep(0.15)  # let cursor blink settle
                frame = await cap.capture_frame()
                image = frame.image
            except Exception as e:
                raise HTTPException(502, f"capture failed: {e}")
            finally:
                try:
                    await cap.close()
                except Exception:
                    pass

        # Optional ROI crop — much better OCR signal on a tight band.
        if y_pct is not None:
            h, _w = image.shape[:2]
            band = max(20, int(req.band_pct * h))
            y_center = int(y_pct * h)
            y0 = max(0, y_center - band)
            y1 = min(h, y_center + band)
            if y1 - y0 >= 16:
                image = image[y0:y1, :]

        from handsneyes.core.vision.imaging import (
            enhance_for_screen,
            numpy_to_base64_png,
            resize_for_mllm,
        )
        b64 = numpy_to_base64_png(
            resize_for_mllm(
                enhance_for_screen(image),
                max_dimension=1280, min_dimension=512,
            )
        )

        cfg = _commander_cfg()
        base_url = getattr(
            cfg, "lmstudio_base_url", "http://localhost:1234/v1",
        )
        model = req.model or "nanonets-ocr-s"

        import httpx as _httpx
        prompt = (
            "You are an OCR system. The image shows a thin horizontal "
            "strip of a computer screen centred on a text input field. "
            "Return ONLY the exact text inside that input field. No "
            "explanation, no quotes, no markdown — just the raw text. "
            "If the field is empty, return an empty string. Do not "
            "include surrounding UI labels or chrome."
        )
        try:
            async with _httpx.AsyncClient(
                base_url=base_url, timeout=30.0,
            ) as client:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                }},
                                {"type": "text", "text": (
                                    "Text in the focused input field:"
                                )},
                            ]},
                        ],
                        "max_tokens": 500,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = (
                    data["choices"][0]["message"]["content"] or ""
                ).strip()
                # Strip surrounding quotes the OCR model sometimes
                # adds even when told not to.
                if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
                    text = text[1:-1]
        except Exception as e:
            raise HTTPException(502, f"OCR call failed: {e}")

        return JSONResponse({
            "ok": True, "text": text, "model": model,
            "roi": (
                {"x_pct": x_pct, "y_pct": y_pct,
                 "band_pct": req.band_pct}
                if y_pct is not None else None
            ),
        })

    @app.post("/api/vault/create")
    async def vault_create(req: VaultCreateRequest) -> JSONResponse:
        """Create a fresh vault file with one seed entry.

        Wipes any existing vault when ``overwrite=True``. Used by the
        cc UI's "Create new vault" tab in the Unlock modal — gives
        operators who forgot the master passphrase a one-click reset
        path without dropping to a terminal.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        from handsneyes.core.vault import DEFAULT_PATH, Vault

        path = DEFAULT_PATH
        # If DEFAULT_PATH resolved to the legacy terminaleyes location
        # (the fallback in vault/__init__.py), we want to create the
        # NEW vault at the handsneyes path so the legacy file is
        # untouched and the next read picks up our fresh one.
        from pathlib import Path as _Path
        handsneyes_dir = _Path.home() / ".config" / "handsneyes"
        path = handsneyes_dir / "vault.enc"

        if path.exists() and not req.overwrite:
            raise HTTPException(
                409,
                f"vault file already exists at {path} — pass "
                f"overwrite=true to replace it",
            )

        # Make sure the legacy terminaleyes vault doesn't shadow our
        # new one (vault loader prefers handsneyes path first, so a
        # leftover legacy file is harmless but we delete it on
        # overwrite for cleanliness).
        legacy = _Path.home() / ".config" / "terminaleyes" / "vault.enc"
        try:
            handsneyes_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as e:
            raise HTTPException(500, f"could not create vault dir: {e}")

        if req.overwrite:
            for p in (path, legacy):
                try:
                    if p.exists():
                        p.unlink()
                except OSError as e:
                    logger.warning("could not remove %s: %s", p, e)

        try:
            vault = Vault(req.passphrase, path=path)
            vault.set(req.entry_name, req.entry_value)
        except Exception as e:
            raise HTTPException(502, f"vault creation failed: {e}")

        return JSONResponse({
            "ok": True,
            "path": str(path),
            "entry_name": req.entry_name,
        })

    @app.post("/api/vault/add")
    async def vault_add(req: VaultAddRequest) -> JSONResponse:
        """Add a new entry to an EXISTING vault.

        Validates the passphrase by opening the vault, then writes
        the new entry alongside whatever else was there. Used by
        the cc UI's "Add to vault" tab in the Unlock modal so an
        operator can store an extra credential without dropping to
        `handsneyes vault add` on the CLI. Existing entries are
        preserved; an entry with the same name is replaced.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        from handsneyes.core.vault import Vault, VaultPassphraseError
        try:
            vault = Vault(req.passphrase)
            # Touch a list/read to force a decrypt — Vault is lazy
            # and a wrong passphrase would otherwise only surface
            # on .set/.save below.
            vault.names()
            vault.set(req.entry_name, req.entry_value)
        except VaultPassphraseError:
            raise HTTPException(401, "vault passphrase rejected")
        except FileNotFoundError:
            raise HTTPException(
                404,
                "no vault file — create one via /api/vault/create first",
            )
        except Exception as e:
            raise HTTPException(502, f"vault add failed: {e}")
        # Cache the verified passphrase for the cc session so
        # subsequent /api/keyboard/from-vault calls don't re-prompt.
        app.state.vault_passphrase = req.passphrase
        return JSONResponse({
            "ok": True,
            "entry_name": req.entry_name,
            "entries": vault.names(),
        })

    @app.post("/api/vault/unlock-session")
    async def vault_unlock_session(
        req: VaultUnlockSessionRequest,
    ) -> JSONResponse:
        """Cache the vault passphrase for the cc process lifetime.

        Validates the passphrase by opening the vault and listing its
        entries. On success, the passphrase is held in app.state and
        used by /api/keyboard/from-vault to type stored values into
        the host. Returns the list of entry names so the operator
        knows which entries are available — never returns values.
        """
        from handsneyes.core.vault import Vault
        try:
            vault = Vault(req.passphrase)
            entries = vault.names()
        except Exception as e:
            raise HTTPException(401, f"vault unlock failed: {e}")
        app.state.vault_passphrase = req.passphrase
        logger.info(
            "vault session unlocked; %d entries available", len(entries),
        )
        return JSONResponse({"ok": True, "entries": entries})

    @app.post("/api/vault/lock-session")
    def vault_lock_session() -> JSONResponse:
        """Forget the cached vault passphrase."""
        app.state.vault_passphrase = None
        return JSONResponse({"ok": True})

    @app.post("/api/keyboard/from-vault")
    async def keyboard_from_vault(
        req: KeyboardFromVaultRequest,
    ) -> JSONResponse:
        """Type a vault entry's value at the host's focused field.

        Lets agentic callers (an LLM driving the cc) hand the host a
        password — typically for a sudo or login prompt — without ever
        seeing the value themselves. The flow is:

          vault file -> kb backend bytes -> Pi BT HID -> host keyboard

        Never returns or logs the actual value, only its length.

        Requires a prior /api/vault/unlock-session call. Returns 401
        if no passphrase is cached.
        """
        if app.state.vault_passphrase is None:
            raise HTTPException(
                401,
                "no vault session — call /api/vault/unlock-session first",
            )
        from handsneyes.core.vault import Vault
        try:
            vault = Vault(app.state.vault_passphrase)
            value = vault.get(req.entry)
        except KeyError:
            raise HTTPException(404, f"vault has no entry {req.entry!r}")
        except Exception as e:
            raise HTTPException(500, f"vault read failed: {e}")

        async def _type(kb):
            try:
                await kb.send_text(value, secret=True)
            except TypeError:
                # Older backend signature without secret kwarg.
                await kb.send_text(value)
            if req.append_enter:
                await kb.send_keystroke("Enter")
            return JSONResponse({
                "ok": True,
                "entry": req.entry,
                "length": len(value),
                "append_enter": req.append_enter,
            })
        return await _with_keyboard(_type)

    @app.get("/api/state")
    def state() -> JSONResponse:
        latest = store.latest()
        active = runner.active()
        cfg = _commander_cfg()
        return JSONResponse({
            "busy": runner.is_busy(),
            "latest_id": latest.id if latest else None,
            "frame_count": store.count(),
            "active_run": active.public() if active else None,
            "screen_width": cfg.screen_width,
            "screen_height": cfg.screen_height,
            # Expose the active target's platform so the SPA can
            # default the platform-dropdown to it. Without this the
            # dropdown always loads at its HTML default ("linux"),
            # and Ctrl-letter shortcuts to a macOS target never get
            # remapped to Cmd-letter — Ctrl-A would not select-all.
            "active_platform": app.state.active_platform,
        })

    # ── homer retrain (online training) ──────────────────────────
    # ── homer slot management ────────────────────────────────────
    # Three slots:
    #   Live      — what the homer loads at runtime (committed to git)
    #   Previous  — one-deep rollback target (outside git, in user data)
    #   Candidate — newly trained, lives in data/ml/checkpoints/pointer_accel-vN
    #
    # Tune: train candidate -> rotate (live -> previous, candidate -> live).
    # Rollback: swap live <-> previous (so a second rollback puts it back).
    from pathlib import Path as _PathHomer
    _HOMER_RUNS_ROOT = _PathHomer.home() / ".local/share/handsneyes/runs"

    def _homer_live_for(platform: str) -> _PathHomer:
        # The installed model for a given platform. Each platform's
        # adapter package has its own models/ directory — Ubuntu's stays
        # untouched when a macOS tune ships, and vice versa.
        return _PathHomer(
            f"src/handsneyes/platforms/{platform}/models/pointer_accel"
        )

    def _homer_prev_for(platform: str) -> _PathHomer:
        # Per-platform Previous slot. Outside the git tree so model
        # state doesn't churn tracked files.
        return _PathHomer.home() / (
            f".local/share/handsneyes/model_slots/{platform}/previous/pointer_accel"
        )

    def _homer_last_tune_ts_path(platform: str) -> _PathHomer:
        return _PathHomer.home() / (
            f".config/handsneyes/last_tune_ts.{platform}"
        )

    def _homer_read_last_tune_ts(platform: str) -> float:
        try:
            return float(_homer_last_tune_ts_path(platform).read_text().strip())
        except Exception:
            return 0.0

    def _homer_write_last_tune_ts(platform: str, ts: float) -> None:
        try:
            p = _homer_last_tune_ts_path(platform)
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            p.write_text(f"{ts}\n")
        except Exception:
            logger.exception("could not persist last_tune_ts")

    def _homer_count_trajectories_since(
        ts: float, platform: str,
    ) -> int:
        """Count homer history.jsonl files newer than ts whose rows are
        tagged with this platform (or untagged — those are legacy
        linux_gnome). Per-platform counter so the macOS Tune button
        shows only macOS clicks and vice versa."""
        if not _HOMER_RUNS_ROOT.exists():
            return 0
        n = 0
        for hist in _HOMER_RUNS_ROOT.glob("**/homer/*/history.jsonl"):
            try:
                if hist.stat().st_mtime < ts:
                    continue
            except OSError:
                continue
            # Sample the first line to read its platform tag (cheap;
            # a whole-trajectory is one platform).
            tagged_platform: str | None = None
            try:
                with hist.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                    if first:
                        try:
                            tagged_platform = (
                                json.loads(first).get("platform")
                            )
                        except json.JSONDecodeError:
                            pass
            except OSError:
                continue
            row_platform = tagged_platform or "linux_gnome"
            if row_platform == platform:
                n += 1
        return n

    def _homer_copy_dir_contents(src: _PathHomer, dst: _PathHomer) -> None:
        """Mirror src/* into dst/ (replacing existing files). Both dirs
        must already exist. Used for slot operations on a directory
        that is a checked-out git tree — we mutate file contents
        in place rather than swapping the dir itself."""
        import shutil as _shutil
        for f in src.iterdir():
            if f.is_file():
                _shutil.copy2(f, dst / f.name)

    def _homer_install_with_rotation(
        candidate: _PathHomer, platform: str,
    ) -> None:
        """Backup live -> previous, then install candidate -> live —
        all scoped to the given platform so other platforms' models
        are untouched."""
        import shutil as _shutil
        live = _homer_live_for(platform)
        prev = _homer_prev_for(platform)
        if live.exists() and any(live.iterdir()):
            prev.parent.mkdir(parents=True, exist_ok=True)
            if prev.exists():
                _shutil.rmtree(prev)
            _shutil.copytree(live, prev)
        live.mkdir(parents=True, exist_ok=True)
        _homer_copy_dir_contents(candidate, live)

    def _homer_swap_live_and_previous(platform: str) -> bool:
        """Swap Live <-> Previous for a single platform. Returns False
        if no previous slot for that platform.

        Implementation note: Live is a git-tracked dir; we mutate
        its files in place rather than replacing the directory itself,
        so other tooling watching that path keeps working."""
        import shutil as _shutil
        live = _homer_live_for(platform)
        prev = _homer_prev_for(platform)
        if not prev.exists():
            return False
        if not live.exists():
            live.mkdir(parents=True, exist_ok=True)
        tmp = prev.parent / "_swap_tmp"
        if tmp.exists():
            _shutil.rmtree(tmp)
        _shutil.copytree(live, tmp)
        for f in list(live.iterdir()):
            if f.is_file():
                f.unlink()
        _homer_copy_dir_contents(prev, live)
        _shutil.rmtree(prev)
        _shutil.copytree(tmp, prev)
        _shutil.rmtree(tmp)
        return True

    @app.get("/api/homer/training-state")
    def homer_training_state() -> JSONResponse:
        platform = app.state.active_platform
        last_tune_ts = _homer_read_last_tune_ts(platform)
        return JSONResponse({
            "platform": platform,
            "n_trajectories_since_train": (
                _homer_count_trajectories_since(last_tune_ts, platform)
            ),
            "last_tune_ts": last_tune_ts,
            "has_previous": _homer_prev_for(platform).exists(),
            "is_retraining": app.state.is_retraining,
            "last_retrain": app.state.last_retrain,
        })

    @app.post("/api/homer/retrain")
    async def homer_retrain() -> JSONResponse:
        """Tune the homer's pointer_accel model on accumulated data.

        Flow: build dataset (excluding bad active-learning rows) →
        warm-start from currently-shipped Live → train → on training
        success, rotate slots (Live → Previous, candidate → Live) and
        persist the tune timestamp so the trajectory counter resets.

        No canary gate — the operator's safety net is the Rollback
        button, which swaps Live ↔ Previous. The trade is: minor model
        regressions might briefly ship before the operator notices and
        rolls back, in exchange for the loop being autonomous.
        """
        if app.state.is_retraining:
            raise HTTPException(409, "retrain already in progress")
        app.state.is_retraining = True
        platform = app.state.active_platform
        async def _run():
            import sys as _sys
            from pathlib import Path as _Path
            summary_path = _Path("/tmp/retrain_summary.json")
            installed = False
            install_error: str | None = None
            try:
                # Pass --platform so retrain_homer.py builds a corpus
                # of just this OS's samples (different acceleration
                # curves between linux_gnome and macos must not mix).
                proc = await asyncio.create_subprocess_exec(
                    _sys.executable, "scripts/retrain_homer.py",
                    "--summary-out", str(summary_path),
                    "--platform", platform,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                import time as _time_local
                from handsneyes.ui.log_bus import (
                    LogEvent as _LogEvent,
                )
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    bus.publish(_LogEvent(
                        ts=_time_local.time(), level="INFO",
                        source="retrain", msg=line, run_id=None,
                    ))
                rc = await proc.wait()
                verdict: dict | None = None
                try:
                    verdict = json.loads(summary_path.read_text())
                except Exception:
                    verdict = None
                # Find the candidate checkpoint dir for pointer_accel.
                # retrain_homer.py either kept it (accepted), or
                # deleted it (rejected by canary / eval_failed).
                candidate: _Path | None = None
                ck_root = _Path("data/ml/checkpoints")
                import re as _re
                pat = _re.compile(r"^pointer_accel-(?:yaru-)?v(\d+)$")
                best_n = -1
                for d in ck_root.glob("pointer_accel-*"):
                    m = pat.match(d.name)
                    if not m or not (d / "config.json").exists():
                        continue
                    n = int(m.group(1))
                    try:
                        if d.stat().st_mtime < _time_local.time() - 3600:
                            continue  # not from this run (older than 1h)
                    except OSError:
                        continue
                    if n > best_n:
                        best_n = n
                        candidate = d
                if rc == 0 and candidate is not None:
                    try:
                        _homer_install_with_rotation(candidate, platform)
                        _homer_write_last_tune_ts(platform, _time_local.time())
                        installed = True
                    except Exception as e:
                        install_error = f"slot rotation failed: {e}"
                        logger.exception("slot rotation failed")
                app.state.last_retrain = {
                    "rc": rc, "verdict": verdict,
                    "platform": platform,
                    "installed": installed,
                    "install_error": install_error,
                    "candidate": candidate.name if candidate else None,
                    "ts": _time_local.time(),
                }
                bus.publish(_LogEvent(
                    ts=_time_local.time(), level="INFO", source="retrain",
                    msg=(
                        f"tune done rc={rc} candidate={candidate.name if candidate else 'none'} "
                        f"installed={installed} "
                        + (f"err={install_error}" if install_error else "")
                    ),
                    run_id=None,
                ))
            except Exception as e:
                logger.exception("retrain subprocess failed")
                import time as _time_local
                app.state.last_retrain = {
                    "rc": -1, "verdict": None, "error": str(e),
                    "installed": False,
                    "ts": _time_local.time(),
                }
            finally:
                app.state.is_retraining = False
        asyncio.create_task(_run())
        return JSONResponse({"ok": True, "started": True})

    @app.post("/api/homer/rollback")
    def homer_rollback() -> JSONResponse:
        """Swap Live <-> Previous slots for the active platform.
        Idempotent under the same previous: clicking Rollback twice
        puts you back where you started. Returns 404 if there is no
        Previous slot for this platform."""
        import time as _time_local
        platform = app.state.active_platform
        if not _homer_swap_live_and_previous(platform):
            raise HTTPException(
                404, f"no previous slot to roll back to for {platform}",
            )
        # Persist a fresh tune timestamp so the trajectory counter
        # resets at the rollback point — clicks accumulated since the
        # bad install should count toward the next tune attempt.
        _homer_write_last_tune_ts(platform, _time_local.time())
        return JSONResponse({"ok": True, "platform": platform})

    # ── capture-source toggle (self capture) ──────────────────────
    # Lets the UI flip between webcam (default) and "grab local
    # display" without editing targets.toml + restarting cc. Takes
    # effect on the NEXT /api/run or /api/mouse/click_at — already-
    # running runs aren't interrupted. CaptureSourceRequest lives at
    # module scope alongside the other body schemas.

    @app.get("/api/capture-source")
    def capture_source_state() -> JSONResponse:
        return JSONResponse({
            "self_capture": bool(
                app.state.runtime_state.get("use_self_capture", False)
            ),
        })

    @app.post("/api/capture-source")
    def capture_source_set(req: CaptureSourceRequest) -> JSONResponse:
        app.state.runtime_state["use_self_capture"] = bool(req.self_capture)
        logger.info(
            "capture source override → %s",
            "screen" if req.self_capture else "(target default)",
        )
        return JSONResponse({"ok": True, "self_capture": req.self_capture})

    # ── pointer-accel scale (cross-target tuning) ─────────────────
    # The pointer_accel model is trained at one effective display
    # resolution (the dev mac's "UI Looks like" size). When deployed
    # against a target running at a different effective resolution
    # — typical when controlling a different mac via screen-share —
    # the same HID moves the same physical pixels but a different
    # PERCENT of the target's screen, so the model's percent-keyed
    # predictions systematically over- or under-shoot by the ratio
    # of effective resolutions. The scale_x / scale_y multipliers
    # below are applied to the model's HID output at send time. A
    # value of 1.0 is no-op; for a remote at half the dev's
    # effective width (e.g. v7 trained at 3840 wide, remote at 1728
    # wide) the right scale is roughly 1728/3840 = 0.45.
    # Schema lives at module scope so Pydantic can fully resolve the
    # type at openapi-generation time (closure-scoped BaseModels
    # produce ForwardRef errors).

    # Pointer-accel scale is persisted to ~/.config/handsneyes/
    # pointer_accel_scale.json so it survives cc restarts. Without
    # this every restart resets to the 1.0/1.0 default and the SPA
    # loads the default until the operator (or a curl POST) sets
    # it again — easy to forget after a restart, with the symptom
    # of "clicks land way off" until you notice the UI shows 1.0.
    from pathlib import Path as _PathScale
    _SCALE_PATH = _PathScale.home() / ".config/handsneyes/pointer_accel_scale.json"

    def _load_persisted_scale() -> tuple[float, float] | None:
        try:
            import json as _json
            with open(_SCALE_PATH) as f:
                d = _json.load(f)
            return float(d["scale_x"]), float(d["scale_y"])
        except Exception:
            return None

    def _save_persisted_scale(sx: float, sy: float) -> None:
        try:
            import json as _json
            _SCALE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(_SCALE_PATH, "w") as f:
                _json.dump({"scale_x": sx, "scale_y": sy}, f)
        except Exception as e:
            logger.warning("could not persist pointer-accel-scale: %s", e)

    # Bootstrap runtime_state from disk if a previous session saved
    # a value. Runs once at server startup (this closure runs at
    # app construction, before any request).
    _persisted = _load_persisted_scale()
    if _persisted is not None:
        app.state.runtime_state["pointer_accel_scale_x"] = _persisted[0]
        app.state.runtime_state["pointer_accel_scale_y"] = _persisted[1]
        logger.info(
            "pointer-accel-scale: loaded persisted x=%.3f y=%.3f",
            _persisted[0], _persisted[1],
        )

    @app.get("/api/pointer-accel-scale")
    def pointer_accel_scale_state() -> JSONResponse:
        return JSONResponse({
            "scale_x": float(
                app.state.runtime_state.get(
                    "pointer_accel_scale_x", 1.0,
                ),
            ),
            "scale_y": float(
                app.state.runtime_state.get(
                    "pointer_accel_scale_y", 1.0,
                ),
            ),
        })

    @app.post("/api/pointer-accel-scale")
    def pointer_accel_scale_set(
        req: PointerAccelScaleRequest,
    ) -> JSONResponse:
        app.state.runtime_state["pointer_accel_scale_x"] = float(req.scale_x)
        app.state.runtime_state["pointer_accel_scale_y"] = float(req.scale_y)
        _save_persisted_scale(float(req.scale_x), float(req.scale_y))
        logger.info(
            "pointer-accel scale override → x=%.3f y=%.3f (persisted)",
            req.scale_x, req.scale_y,
        )
        return JSONResponse({
            "ok": True,
            "scale_x": req.scale_x,
            "scale_y": req.scale_y,
        })

    # ── scheduler (recurring controller intents) ──────────────────
    # The chat-form's Send button fires a controller intent once. The
    # scheduler fires the same intent on an interval — "every 5 min,
    # check the build status" kind of thing. On a tick where the
    # runner is busy (manual click in flight, tune retraining, etc.)
    # the tick is silently skipped; the next fire is still one interval
    # away from the previous SCHEDULED tick, not the skipped one, so
    # cadence drift is bounded.
    def _public_job(job: dict) -> dict:
        return {
            k: v for k, v in job.items()
            if k not in ("task", "options")  # task isn't JSON; options have secrets
        } | {
            # Mirror non-secret options so the UI can show the dry-run /
            # platform / no-focus a job was created with.
            "options": {
                k: v for k, v in job.get("options", {}).items()
                if k not in ("password", "vault_passphrase")
            },
        }

    async def _scheduler_loop(job_id: str) -> None:
        import time as _time_local
        job = app.state.schedules.get(job_id)
        if job is None:
            return
        interval_s = job["interval_minutes"] * 60.0
        # First-fire: either now (fire_immediately) or after one interval.
        if job["fire_immediately"]:
            await _scheduler_fire_once(job_id)
        # Anchor to absolute monotonic schedule so that a skipped tick
        # doesn't shift the cadence. The loop computes the next fire
        # time from the original anchor, not from "now".
        anchor = job["created_at"]
        n = 1 if not job["fire_immediately"] else 1
        while job_id in app.state.schedules:
            target = anchor + n * interval_s
            now = _time_local.time()
            wait = target - now
            if wait > 0:
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    return
            if job_id not in app.state.schedules:
                return
            await _scheduler_fire_once(job_id)
            n += 1

    async def _scheduler_fire_once(job_id: str) -> None:
        import time as _time_local
        job = app.state.schedules.get(job_id)
        if job is None:
            return
        if runner.is_busy():
            job["skipped_count"] = job.get("skipped_count", 0) + 1
            job["last_skipped_at"] = _time_local.time()
            bus.publish(LogEvent(
                ts=_time_local.time(), level="INFO", source="scheduler",
                msg=f"[{job_id}] tick skipped — runner busy",
                run_id=None,
            ))
            return
        opts = job["options"]
        try:
            rec = await runner.start(
                intent=job["intent"],
                no_focus=opts.get("no_focus", False),
                vault=opts.get("vault"),
                password=opts.get("password"),
                vault_passphrase=opts.get("vault_passphrase"),
                skip_verify=opts.get("skip_verify", False),
                platform=opts.get("platform", "linux"),
                dry_run=opts.get("dry_run", False),
                allow_llm_fallback=opts.get("allow_llm_fallback", True),
                planner=opts.get("planner", "auto"),
                ml_adapter=opts.get("ml_adapter"),
            )
            job["last_run_id"] = rec.run_id
            job["last_fire_at"] = _time_local.time()
            job["fire_count"] = job.get("fire_count", 0) + 1
            bus.publish(LogEvent(
                ts=_time_local.time(), level="INFO", source="scheduler",
                msg=f"[{job_id}] fired run {rec.run_id} for intent {job['intent']!r}",
                run_id=rec.run_id,
            ))
        except Exception as e:
            job["last_error"] = str(e)
            job["last_error_at"] = _time_local.time()
            logger.exception("scheduler[%s] fire failed", job_id)

    @app.post("/api/scheduler/create")
    async def scheduler_create(req: ScheduleCreateRequest) -> JSONResponse:
        import time as _time_local
        import uuid as _uuid
        job_id = _uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "intent": req.intent,
            "interval_minutes": req.interval_minutes,
            "created_at": _time_local.time(),
            "fire_immediately": req.fire_immediately,
            "fire_count": 0,
            "skipped_count": 0,
            "last_fire_at": None,
            "last_skipped_at": None,
            "last_run_id": None,
            "last_error": None,
            "options": {
                "no_focus": req.no_focus,
                "vault": req.vault,
                "password": req.password,
                "vault_passphrase": req.vault_passphrase,
                "skip_verify": req.skip_verify,
                "platform": req.platform,
                "dry_run": req.dry_run,
                "allow_llm_fallback": req.allow_llm_fallback,
                "planner": req.planner,
                "ml_adapter": req.ml_adapter,
            },
        }
        app.state.schedules[job_id] = job
        task = asyncio.create_task(
            _scheduler_loop(job_id), name=f"schedule-{job_id}",
        )
        job["task"] = task
        return JSONResponse({"ok": True, **_public_job(job)})

    @app.get("/api/scheduler/list")
    def scheduler_list() -> JSONResponse:
        return JSONResponse({
            "jobs": [_public_job(j) for j in app.state.schedules.values()],
        })

    @app.post("/api/scheduler/cancel")
    async def scheduler_cancel(req: ScheduleCancelRequest) -> JSONResponse:
        job = app.state.schedules.pop(req.id, None)
        if job is None:
            raise HTTPException(404, f"no such schedule {req.id!r}")
        task = job.get("task")
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        return JSONResponse({"ok": True, "id": req.id})

    @app.on_event("shutdown")
    async def _scheduler_shutdown() -> None:
        for job in list(app.state.schedules.values()):
            t = job.get("task")
            if t is not None:
                t.cancel()
        app.state.schedules.clear()

    return app
