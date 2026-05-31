# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commander; lint cleanup deferred.
"""Visual-servo cursor homer.

Replaces the open-loop ``closed_loop_homer`` whose cursor position was
dead-reckoned from a fixed HID-per-percent constant baked from a prior
target Mac. On a different machine the constant lies and the cursor
never goes where the homer thinks it goes.

This homer instead **sees the cursor**:

1. Slam to the top-left corner (known coarse start).
2. **Calibrate** by sending a known burst horizontally then vertically;
   diff before/after; the moving blob IS the cursor and its travel
   gives a real per-session ``pct_per_hid`` ratio.
3. **Visual servo**: each step locates the target via ShowUI and the
   cursor via frame-diff against the previous frame. We send a move
   proportional to the residual, then re-detect the cursor in the
   post-move frame and refine the HID ratio online.
4. **Geometric click gate**: click iff the visually-detected cursor
   sits within ``CLICK_TOL_PCT`` of the ShowUI target for two
   consecutive frames. No "ask gemma to confirm" rubber stamp.

Reuses scene-map + ShowUI grounding from ``ClosedLoopHomer`` for the
*what* (target), but replaces the *where* (cursor) entirely.
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from handsneyes.core.vision.closed_loop_homer import (
    ClosedLoopHomer,
)
from handsneyes.core.vision.cursor_finder import (
    CursorHit,
    annotate_cursor,
    find_cursor_by_variance,
    find_cursor_hsv,
    find_cursor_hsv_motion,
    find_cursor_hsv_motion_directed,
    find_cursor_hsv_near,
    setup_instructions,
)
from handsneyes.core.vision.ocr_finder import (
    annotate_ocr_hit,
    find_text as ocr_find_text,
    have_ocr,
)
from handsneyes.core.vision.imaging import (
    enhance_for_screen,
    numpy_to_base64_png,
    resize_for_mllm,
)

if TYPE_CHECKING:
    # InteractiveSession typing-only import dropped — Phase A AgentContext
    # plus session-adapter shims provide the interface the homer uses.
    InteractiveSession = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ───────────── tunables ─────────────

# Per-session calibration burst sizes. Big enough that cursor motion is
# unmistakable on a webcam view, small enough not to clip the screen.
CALIB_BURST_HID = 180

# How tolerant of vector mismatch we are when picking the cursor blob
# from a diff. ``0.6`` means the blob's motion vector must roughly
# share a hemisphere with the expected motion (cosine ≥ 0).
DIRECTION_AGREEMENT_MIN = 0.0

# Click gate: visually-detected cursor must be within this fraction of
# the image (in both axes) of the ShowUI target for ``CONFIRM_FRAMES``
# consecutive frames.
CLICK_TOL_PCT = 0.012
CONFIRM_FRAMES = 2

# After the first click misses, retry up to this many times, nudging
# the cursor through a small diamond pattern around the aim point.
CLICK_RETRY_PATTERN_HID: list[tuple[int, int]] = [
    (0, -10),    # up
    (0, +10),    # down (counter-act prev)
    (-10, 0),    # left
    (+10, 0),    # right
    (0, -10),    # up again (fine vertical)
]

# Cap each move at this fraction of remaining residual to avoid
# overshoot when ``pct_per_hid`` is still being learned.
STEP_DISTANCE_FRACTION = 0.55

# Don't send moves smaller than this — sub-threshold HID often gets
# eaten by cursor acceleration with no observable motion.
MIN_HID_PER_AXIS = 4

# Hard ceiling on HID per axis per step — prevents runaway moves when
# the learned ratio mis-collapses.
MAX_HID_PER_AXIS = 220

# Floor / ceiling on the learned HID-to-image ratio. Wide bounds so
# the homer can adapt to remote targets running at different effective
# resolutions / acceleration curves (e.g. controlling an MBP 14" at
# "Looks like 1728×1117" from a dev mac at "Looks like 3840×2160",
# where the remote's actual ratio sits around 0.1‰ — well below the
# previous 0.6‰ floor that locked the homer to its default seed).
RATIO_MIN = 0.00005
RATIO_MAX = 0.0050

# Settle delay between "send move" and "capture post-move frame".
SETTLE_SEC = 0.18

# Diff threshold and morphology for cursor extraction.
DIFF_THRESH = 22
DIFF_DILATE_KERNEL = 5

# Blob area filter (fraction of image area).
BLOB_MIN_AREA = 0.00003   # ~30px on a 1280×720 frame
BLOB_MAX_AREA = 0.020     # don't latch onto a UI repaint

# EMA smoothing on the learned ratio.
RATIO_EMA = 0.5

# Default initial ratio guess (refined on every move).
DEFAULT_PCT_PER_HID = 1.6 / 1920.0

# Cursor hotspot offset: when we visually detect the cursor (via HSV
# or variance), the centroid we measure sits roughly half the cursor
# size DOWN-RIGHT of the hotspot (default arrow points up-left, hotspot
# at the tip). To make the click land on the target, we aim the
# centroid at ``target + HOTSPOT_OFFSET`` so the hotspot ends up on
# the original target. In practice the cursor also overshoots aim
# slightly so a small offset is sufficient.
HOTSPOT_OFFSET_X_PCT = 0.005
HOTSPOT_OFFSET_Y_PCT = 0.005

MAX_STEPS = 30
PROOF_DIR = Path("/tmp/handsneyes_homer")


@dataclass
class StepRecord:
    cursor_img: tuple[float, float] | None
    target_img: tuple[float, float] | None
    residual_pct: float | None
    hid_dx: int
    hid_dy: int
    measured_dx_pct: float | None = None
    measured_dy_pct: float | None = None
    ratio_x: float | None = None
    ratio_y: float | None = None
    note: str = ""


def _persist_step(
    run_dir: Path, record: "StepRecord", *, platform: str | None = None,
) -> None:
    """Append one step record as a JSONL row to
    ``<run_dir>/history.jsonl``. Best-effort; errors are swallowed
    so a logging failure can't break a live homer run.

    Each row is ``{ts, platform, hid_dx, hid_dy, cursor_img: [x, y] | null,
    measured_dx_pct, measured_dy_pct, ratio_x, ratio_y, ...}`` — the
    fields a forward-model trainer needs to learn the OS-side
    pointer-acceleration curve from observed cursor deltas.

    ``platform`` tags the row with the active target's adapter name
    (e.g. "linux_gnome", "macos"). The dataset builder filters by
    this so a per-platform retrain doesn't mix libinput-adaptive and
    IOHID samples — those are fundamentally different curves and
    mixing them produces a model worse than either pure subset.
    Rows written before this field existed are absent it; the
    builder treats untagged rows as "linux_gnome" for back-compat
    (every existing row is from the Yaru/Ubuntu target).
    """
    try:
        row = asdict(record)
        # ``asdict`` turns tuples into lists, which is fine for JSON.
        row["ts"] = time.time()
        if platform:
            row["platform"] = platform
        with (run_dir / "history.jsonl").open(
            "a", encoding="utf-8",
        ) as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Step recording must never break a live run.
        pass


def _record_step(
    run_dir: Path,
    history: list["StepRecord"],
    record: "StepRecord",
    *,
    platform: str | None = None,
) -> None:
    """Append ``record`` to in-memory ``history`` *and* persist it.

    Used at every ``history.append(StepRecord(...))`` site so a
    single API change covers both the existing in-memory contract
    and the new on-disk JSONL training log. The optional ``platform``
    tag flows through to the JSONL row so the dataset builder can
    train per-OS corpora.
    """
    history.append(record)
    _persist_step(run_dir, record, platform=platform)


_POINTER_ACCEL_CHECKPOINT_CANDIDATES = (
    # v5: trained on HSV-tracked measurements (per-step finder is
    # the position-aware ``find_cursor_hsv_near`` so the per-row
    # delta is pixel-accurate instead of frame-diff-noisy).
    # v4: redglass cursor at size 96, frame-diff measured (v4 was
    # the data ceiling that motivated the HSV cross-check fix).
    # v3+: direct inverse (single forward pass at runtime).
    # v1/v2: legacy forward models that the runtime Newton-inverts.
    # Platform-bundled (handsneyes Phase B): ships with each adapter.
    Path(__file__).resolve().parent.parent.parent
        / "platforms" / "linux_gnome" / "models" / "pointer_accel",
    # Legacy terminaleyes paths (for back-compat with users who
    # retrained against the old data/ml layout).
    Path("data/ml/checkpoints/pointer_accel-v5"),
    Path("data/ml/checkpoints/pointer_accel-v4"),
    Path("data/ml/checkpoints/pointer_accel-v3"),
    Path("data/ml/checkpoints/pointer_accel-v2"),
    Path("data/ml/checkpoints/pointer_accel-v1"),
)

# Long-jump model: predicts the TOTAL HID for a full slam-to-target
# move. Used at the top of home_to_pixel to fire a chain of back-to-
# back bursts (no captures between) that lands the cursor near the
# target in one shot, before handing off to the standard closed-loop
# servo for the small residual.
_LONGJUMP_CHECKPOINT_CANDIDATES = (
    # Platform-bundled (handsneyes Phase B).
    Path(__file__).resolve().parent.parent.parent
        / "platforms" / "linux_gnome" / "models" / "longjump",
    # Legacy terminaleyes paths.
    Path("data/ml/checkpoints/longjump-v2"),
    Path("data/ml/checkpoints/longjump-v1"),
)

# Per-checkpoint runtime calibration knob. The chained-burst
# pattern triggers Mac's pointer-acceleration curve differently
# than the slow-paced training trajectories, so v1 sometimes over-
# and sometimes under-shoots by 15-25% in a single sample. A flat
# scalar didn't help consistently (one click landed tighter, the
# next over-corrected the opposite way), so v1 is left at 1.0 and
# the post-burst HSV cascade + closed-loop refinement absorbs the
# residual. v2+ is the principled fix — retrain on data captured
# under the actual chained-burst runtime.
_LONGJUMP_CALIBRATION = {
    "longjump-v1": (1.0, 1.0),
    "longjump-v2": (1.0, 1.0),
}


def _try_load_longjump():
    """Best-effort load of the trained long-jump model. Returns None
    if the package or checkpoint is missing — the homer falls back
    to closed-loop-only behaviour (no long-distance HID seed; the
    closed loop chips at the residual one step at a time, taking
    several more iterations per click)."""
    try:
        from handsneyes.core.vision.longjump import LongJumpModel
    except Exception as e:
        logger.debug("longjump module unavailable: %s", e)
        return None
    return None  # sentinel; replaced by _try_load_longjump_for(adapter) below


def _try_load_longjump_for(platform_adapter):
    """Per-platform long-jump loader. Same platform-mismatch protection
    as _try_load_pointer_accel: when the adapter has no model shipped,
    don't substitute a different platform's checkpoint."""
    try:
        from handsneyes.core.vision.longjump import LongJumpModel
    except Exception as e:
        logger.debug("longjump module unavailable: %s", e)
        return None
    candidates: list[Path] = []
    if platform_adapter is not None and hasattr(
        platform_adapter, "longjump_checkpoint",
    ):
        adapter_ckpt = platform_adapter.longjump_checkpoint()
        if adapter_ckpt is not None:
            candidates.append(Path(adapter_ckpt))
        else:
            plat_name = getattr(platform_adapter, "name", "<unknown>")
            logger.info(
                "VisualServoHomer: platform %r has no long-jump "
                "checkpoint shipped. Slam-to-target first step will "
                "use ratio-only open-loop seed (slower).",
                plat_name,
            )
            return None
    else:
        candidates.extend(_LONGJUMP_CHECKPOINT_CANDIDATES)

    for cand in candidates:
        if (cand / "config.json").exists():
            try:
                m = LongJumpModel(cand)
                cal = _LONGJUMP_CALIBRATION.get(cand.name, (1.0, 1.0))
                m._calibration = cal
                logger.info(
                    "VisualServoHomer: loaded long-jump seed model "
                    "from %s (calibration=%s)", cand, cal,
                )
                return m
            except Exception as e:
                logger.warning(
                    "longjump model at %s failed to load: %s",
                    cand, e,
                )
    return None


def _try_load_pointer_accel(platform_adapter=None):
    """Best-effort load of the trained open-loop pointer-accel MLP.

    Resolution order:
      1. ``platform_adapter.pointer_accel_checkpoint()`` — the right
         per-OS checkpoint as declared by the active platform.
      2. The shipped linux_gnome path (legacy fallback for callers
         that don't pass a platform adapter).
      3. ``data/ml/checkpoints/pointer_accel-vN`` — legacy retrain
         output, kept for back-compat.

    Critically: when the platform adapter explicitly returns ``None``
    (e.g. the macos adapter when no macos checkpoint is shipped), we
    DO NOT fall through to the linux_gnome path. Loading a Linux-
    libinput-trained model on a macOS IOHID target makes every click
    wildly off — different acceleration curves entirely. Better to
    have no seed model and let the closed-loop servo refine from
    ratio-only than to seed with a wrong-platform model.
    """
    try:
        from handsneyes.core.vision.pointer_accel import PointerAccelModel
    except Exception as e:
        logger.debug("pointer-accel module unavailable: %s", e)
        return None

    candidates: list[Path] = []
    if platform_adapter is not None and hasattr(
        platform_adapter, "pointer_accel_checkpoint",
    ):
        adapter_ckpt = platform_adapter.pointer_accel_checkpoint()
        if adapter_ckpt is not None:
            candidates.append(Path(adapter_ckpt))
        else:
            # Adapter says: no model for this platform. Don't load a
            # different platform's model by accident.
            plat_name = getattr(platform_adapter, "name", "<unknown>")
            logger.info(
                "VisualServoHomer: platform %r has no pointer-accel "
                "checkpoint shipped. Falling back to ratio-only seed "
                "— the closed-loop servo will refine from there, but "
                "first-iteration error will be larger. Train a model "
                "for this platform via the cc's Tune button or run "
                "scripts/train_pointer_accel.py against the new corpus.",
                plat_name,
            )
            return None
    else:
        # Legacy callers (no adapter passed): keep old behaviour so
        # tests that construct a bare homer still find a model.
        candidates.extend(_POINTER_ACCEL_CHECKPOINT_CANDIDATES)

    for cand in candidates:
        if (cand / "config.json").exists():
            try:
                m = PointerAccelModel(cand)
                logger.info(
                    "VisualServoHomer: loaded pointer-accel seed model "
                    "from %s",
                    cand,
                )
                return m
            except Exception as e:
                logger.warning(
                    "pointer-accel model at %s failed to load: %s",
                    cand, e,
                )
    logger.warning(
        "VisualServoHomer: no pointer-accel checkpoint at %s. The "
        "homer will work but convergence is slower.",
        ", ".join(str(c) for c in candidates),
    )
    return None


@dataclass
class ClickOutcome:
    clicked: bool
    steps: int
    reason: str
    proof_path: str | None = None
    history: list[StepRecord] = field(default_factory=list)


class VisualServoHomer:
    """Closed-loop visual servo homer.

    Cursor position is *measured* every step via frame-diff. The
    HID-to-image ratio is learned online. Models locate the target
    only; geometry decides the click.
    """

    def __init__(self, *, session: "InteractiveSession") -> None:
        self._session = session
        self._pct_per_hid_x = DEFAULT_PCT_PER_HID
        self._pct_per_hid_y = DEFAULT_PCT_PER_HID
        # Observed cursor footprint, in image-area-percent. Updated
        # whenever the HSV finder reports a valid CursorHit. Used to
        # adaptively size the hotspot offset (the centroid-to-hotspot
        # distance scales with cursor pixel size; a 32-px default
        # cursor needs a ~0.5% offset, a 96-px high-contrast cursor
        # needs ~3-5%). Without this, callers with a non-default
        # cursor size see every click land down-left of where they
        # clicked because the static HOTSPOT_OFFSET_*_PCT constants
        # under-estimate the offset for any cursor larger than the
        # default.
        self._cursor_area_pct: float | None = None
        # Hotspot offset calibrated by ``_calibrate_hotspot_at_corner``
        # — measured precisely once per session by exploiting macOS's
        # edge clamp (which clamps the cursor's HOTSPOT, not its
        # centroid, at (0, 0)). When set, this is THE source of truth
        # for the hotspot offset and overrides both the static
        # HOTSPOT_OFFSET_*_PCT constants and the area-derived
        # estimate. Works for any cursor shape (arrow, I-beam, hand,
        # crosshair) without needing to recognise the shape.
        self._calibrated_hotspot_offset: tuple[float, float] | None = None
        # Cruise-mode (fast-send) ratios. Seeded LOW (= chunked
        # default) so the first few cruise bursts hit the
        # CRUISE_MAX_HID cap. A capped burst moves the cursor far
        # enough on screen that find_cursor_hsv_motion gets a
        # robust differential signal (a tiny burst's pre/post
        # frames are nearly identical and the motion-diff blob is
        # too small to meet area filters). HSV-measured deltas
        # then EMA-refine the ratio UP toward whatever the host's
        # actual burst-mode accel response is.
        self._pct_per_hid_fast_x = DEFAULT_PCT_PER_HID
        self._pct_per_hid_fast_y = DEFAULT_PCT_PER_HID
        self._diff_misses_in_a_row = 0
        self._zoom_levels_applied = 0
        # If init's HSV candidate fails motion verification, disable
        # HSV for the rest of the run — there's a static red element
        # (Reddit logo, etc.) that would otherwise hijack every step's
        # detection and freeze the cursor estimate.
        self._hsv_enabled = False
        # Cursor template captured at the first successful initial
        # detection of a run. Subsequent per-step locates inside
        # _servo_loop try this first via ROI-restricted normalised
        # cross-correlation — pixel-precise, sub-millisecond, and
        # robust to the screen-share encoder noise that defeats
        # frame-diff on cross-mac targets. None when no successful
        # initial detection has happened yet (or when the template
        # could not be cropped, e.g. cursor sat too close to the
        # frame edge).
        # Number of consecutive servo-loop iterations where the
        # template-match returned the same pixel position despite
        # the homer having sent a non-zero HID. Counts toward the
        # wedge-detection threshold; reset on any iteration where
        # the template's reported position actually changes.
        self._tm_wedge_count: int = 0
        # Reuse the scene-map and target-keyword machinery from the
        # earlier homer — that part still works, the bug was elsewhere.
        self._helper = ClosedLoopHomer(session=session)
        # Resolve the platform adapter from the session's AgentContext.
        # The adapter declares which pointer_accel + longjump checkpoints
        # to load — Linux libinput and macOS IOHID have different
        # acceleration curves, so a checkpoint trained on one is wrong
        # for the other (user-visible: cursor lands way off on macOS).
        platform_adapter = None
        ctx = getattr(self._session, "_ctx", None)
        if ctx is not None:
            platform_adapter = getattr(ctx, "platform", None)
        # Platform name stamped onto every history.jsonl row so the
        # per-OS retrain corpora stay separate. None for legacy callers
        # that don't supply an adapter (tests / standalone scripts).
        self._platform_name = (
            getattr(platform_adapter, "name", None)
            if platform_adapter is not None else None
        )

        # Optional open-loop pointer-acceleration model. Trained
        # offline (scripts/train_pointer_accel.py); used only as the
        # first-iteration seed of the servo loop. Closed-loop still
        # owns iterations 2+, so absence is harmless.
        self._pointer_accel = _try_load_pointer_accel(platform_adapter)
        # Optional long-jump model for the slam-to-target first move.
        # Trained on aggregated trajectory data
        # (scripts/build_longjump_dataset.py + train_longjump.py).
        # When present, home_to_pixel fires a chain of back-to-back
        # HID bursts based on this model's prediction before entering
        # the closed-loop servo, cutting typical click time ~2.5x.
        self._longjump = _try_load_longjump_for(platform_adapter)

    async def run(self, target_desc: str, button: str = "left") -> ClickOutcome:
        try:
            return await self._run_inner(target_desc, button)
        finally:
            # Reset any browser zoom we applied so we don't leave the
            # page in an unusual state for the next run.
            if self._zoom_levels_applied > 0:
                try:
                    await self._session._executor._keyboard.send_key_combo(
                        ["ctrl"], "0",
                    )
                    print(f"  Zoom reset (was at +{self._zoom_levels_applied})")
                except Exception:
                    pass

    async def _run_inner(self, target_desc: str, button: str) -> ClickOutcome:
        # Honour an AgentContext-supplied output_dir if the session
        # adapter exposes one; otherwise fall back to the legacy
        # per-run dump under /tmp.
        session_out = getattr(self._session, "output_dir", None)
        ts = datetime.now().strftime("%H%M%S_vs")
        if session_out is not None:
            run_dir = session_out / "homer" / ts
        else:
            run_dir = PROOF_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        history: list[StepRecord] = []
        last_proof: str | None = None

        print(f"  Homing (visual servo, HSV): {target_desc}")
        print(f"  Step log: {run_dir}/")

        # 1) Move cursor to a visible position via slam, then nudge
        # back into the screen so the cursor is not pinned in a
        # corner where HSV may struggle near the edge. BUT FIRST,
        # while the cursor is still pinned at the corner, calibrate
        # the hotspot offset (see _calibrate_hotspot_at_corner for
        # the geometry): macOS clamps the cursor's HOTSPOT (not its
        # centroid) at the screen edge, so the centroid we detect
        # at the corner IS the hotspot-to-centroid vector for the
        # current cursor shape.
        await self._slam_to_corner()
        await self._send_hid(40, 40)   # nudge ~40 px diagonally inward
        await asyncio.sleep(0.30)

        # 2) Locate the cursor. Try HSV first (fast, zero-cost) but
        # VERIFY the candidate by sending a nudge and checking it
        # actually moves — otherwise we'd lock onto static red UI
        # elements (Reddit logo, etc.). Fall back to oscillation
        # variance detection if HSV's candidate is static.
        f0 = await self._capture_color()
        cursor_hit = find_cursor_hsv(f0)
        cursor_img: tuple[float, float] | None = None
        if cursor_hit is not None:
            verified = await self._verify_hsv_by_motion(
                (cursor_hit.x_pct, cursor_hit.y_pct), run_dir,
            )
            if verified is not None:
                cursor_img = verified
                self._hsv_enabled = True
                # Record the cursor's screen footprint so the hotspot
                # offset adapts to whatever cursor size the host is
                # actually using.
                self._cursor_area_pct = cursor_hit.area_pct
                print(
                    f"  Cursor (HSV verified): ({cursor_img[0]:.2%}, "
                    f"{cursor_img[1]:.2%}) area={cursor_hit.area_pct*100:.3f}%"
                )
                try:
                    cv2.imwrite(
                        str(run_dir / "00_cursor_init.png"),
                        annotate_cursor(f0, cursor_hit),
                    )
                except Exception:
                    pass
            else:
                print(
                    f"  HSV candidate at ({cursor_hit.x_pct:.2%},"
                    f"{cursor_hit.y_pct:.2%}) did NOT move when nudged — "
                    f"static red element, ignoring."
                )

        if cursor_img is None:
            print("  Using oscillation-variance to find cursor.")
            osc_hit = await self._find_cursor_via_oscillation(run_dir)
            if osc_hit is None:
                print("  Could not detect cursor by oscillation.")
                print(setup_instructions())
                return ClickOutcome(
                    clicked=False, steps=0, reason="cursor_not_found",
                    proof_path=None, history=history,
                )
            cursor_img = osc_hit
            print(
                f"  Cursor (oscillation): ({cursor_img[0]:.2%}, "
                f"{cursor_img[1]:.2%})"
            )
            f0 = await self._capture_color()
            # HSV cross-check using the differential red-mask method.
            # We send a small verification nudge and look for the
            # blob that BECAME red between the pre and post frames —
            # static red UI cancels out, so a hit near the osc result
            # is provably the cursor.
            try:
                pre_check = await self._capture_color()
                # 80 HID ≈ 6% image ≈ 128 px → comfortably bigger
                # than the cursor's own footprint (~30 px wide), so
                # pre and post cursor positions don't overlap and
                # the dilated-pre subtraction leaves a clean
                # "newly red" blob. 20 HID isn't enough — the
                # cursor in pre and post overlaps by more than the
                # 4-pixel dilate buffer, eating most of the signal.
                pre_pct = self._pct_per_hid_x or DEFAULT_PCT_PER_HID
                nudge_hid = 80
                await self._send_hid(nudge_hid, 0)
                await asyncio.sleep(SETTLE_SEC)
                post_check = await self._capture_color()
                hsv_check_hit = find_cursor_hsv_motion_directed(
                    pre_check, post_check,
                    cursor_pre_pct=cursor_img,
                    expected_motion_pct=(nudge_hid * pre_pct, 0.0),
                    max_dist_pct=0.12,
                )
                if hsv_check_hit is None:
                    logger.info(
                        "HSV cross-check: no motion-diff red blob "
                        "near osc result; staying on frame-diff."
                    )
                else:
                    self._hsv_enabled = True
                    cursor_img = (
                        hsv_check_hit.x_pct, hsv_check_hit.y_pct,
                    )
                    logger.info(
                        "HSV cross-check OK (motion-diff) — enabling "
                        f"HSV for per-step measurement; adopted "
                        f"({cursor_img[0]:.2%},{cursor_img[1]:.2%})."
                    )
            except Exception as e:
                logger.warning("HSV cross-check error: %s", e)

        # Stash a cursor-template patch from the initial frame for
        # per-step matching in the servo loop. Both the HSV-verified
        # and the oscillation-only branches converge here with
        # cursor_img + f0 already validated.

        # 3) Locate the target (model: ShowUI via scene-map). One-shot.
        target_img = await self._locate_target(f0, target_desc, run_dir)
        if target_img is None:
            print("  Could not locate target. Abort, no click.")
            return ClickOutcome(
                clicked=False, steps=0, reason="target_lost",
                proof_path=None, history=history,
            )
        # Apply hotspot offset: aim the cursor *centroid* slightly
        # past the link so the cursor *tip* lands on the link.
        hx, hy = self._hotspot_offset_pct()
        target_aim = (
            target_img[0] + hx,
            target_img[1] + hy,
        )
        print(
            f"  Target ≈ ({target_img[0]:.2%}, {target_img[1]:.2%}); "
            f"aim centroid → ({target_aim[0]:.2%}, {target_aim[1]:.2%}) "
            f"to compensate for hotspot offset"
        )

        return await self._servo_loop(
            target_aim=target_aim, target_img=target_img,
            cursor_img=cursor_img, button=button, run_dir=run_dir,
            history=history, target_desc=target_desc,
            verify_navigation=True, last_proof=last_proof,
            confirm_frames=CONFIRM_FRAMES,
            click_tol_pct=CLICK_TOL_PCT,
        )

    async def _servo_loop(
        self, *,
        target_aim: tuple[float, float],
        target_img: tuple[float, float],
        cursor_img: tuple[float, float],
        button: str,
        run_dir: Path,
        history: list[StepRecord],
        target_desc: str,
        verify_navigation: bool,
        last_proof: str | None,
        confirm_frames: int = CONFIRM_FRAMES,
        click_tol_pct: float = CLICK_TOL_PCT,
        click: bool = True,
        axis_aligned: bool = False,
        dragging: bool = False,
        click_count: int = 1,
    ) -> ClickOutcome:
        """Shim to the ROI-stack visual servo. The legacy proportional-
        HID + cascade-detection loop that previously lived here has
        been deleted (see git history); this method exists so external
        callers (run, _run_inner, home_to_pixel) keep the same entry
        point."""
        return await self._servo_loop_roi(
            target_aim=target_aim, target_img=target_img,
            cursor_img=cursor_img, button=button, run_dir=run_dir,
            history=history, target_desc=target_desc,
            verify_navigation=verify_navigation,
            last_proof=last_proof,
            confirm_frames=confirm_frames,
            click_tol_pct=click_tol_pct,
            click=click, axis_aligned=axis_aligned,
            dragging=dragging, click_count=click_count,
        )

    # ────────────────────── manual click-to-pixel ──────────────────────

    async def _home_to_pixel_via_oracle(
        self,
        *,
        x_pct: float,
        y_pct: float,
        button: str,
        hotspot_offset: bool,
        click: bool,
        run_dir: Path,
    ) -> ClickOutcome:
        """Direct-cursor homing loop. Used when the session has a
        cursor oracle (Quartz on macOS self-capture) instead of relying
        on CV against a captured frame.

        Mirrors ``scripts/canary_macos_direct.py``: pre-burst when the
        residual exceeds one HID step, then hand to the pointer-accel
        model for fine adjustment. No slam-to-corner, no HSV / variance
        / frame-diff — none of those produce useful signal when the
        cursor isn't in the captured frame to begin with.
        """
        history: list[StepRecord] = []
        print(
            f"  Homing via cursor oracle → ({x_pct:.2%}, {y_pct:.2%}) "
            f"button={button}"
        )
        print(f"  Step log: {run_dir}/")

        if hotspot_offset:
            hx, hy = self._hotspot_offset_pct()
        else:
            hx = hy = 0.0
        target_aim = (x_pct + hx, y_pct + hy)

        # Per-HID gain on macOS at hid=127. Stays in the file so the
        # canary and the homer don't drift apart — copy if you tune it
        # somewhere else.
        PER_HID_PCT = 0.036
        BIG_DELTA = 0.05
        TOL = 0.008          # ≈ 30 px on 3840 — same as the canary
        MAX_ITERS = 12
        MAX_BURSTS_PER_ITER = 6

        last_pos: tuple[float, float] | None = None
        for it in range(MAX_ITERS):
            pos = await self._session.read_cursor_pct()
            if pos is None:
                # Oracle failed mid-run. Bail out — falling back to
                # CV is pointless on self-capture.
                return ClickOutcome(
                    clicked=False, steps=it,
                    reason="cursor_reader_returned_none",
                    proof_path=None, history=history,
                )
            cx, cy = pos
            dx_pct = target_aim[0] - cx
            dy_pct = target_aim[1] - cy
            if abs(dx_pct) < TOL and abs(dy_pct) < TOL:
                last_pos = (cx, cy)
                print(
                    f"  Converged in {it} iter(s); cursor at "
                    f"({cx:.2%}, {cy:.2%})"
                )
                break

            if abs(dx_pct) > BIG_DELTA or abs(dy_pct) > BIG_DELTA:
                # Pre-burst phase: a single int8 HID maxes out at
                # PER_HID_PCT — single-shot can't cover a large
                # residual within the iter budget. Burst until close
                # enough for the model to take over.
                sign_x = 1 if dx_pct >= 0 else -1
                sign_y = 1 if dy_pct >= 0 else -1
                for _ in range(MAX_BURSTS_PER_ITER):
                    hid_dx_b = sign_x * 127 if abs(dx_pct) > 0.018 else 0
                    hid_dy_b = sign_y * 127 if abs(dy_pct) > 0.018 else 0
                    if hid_dx_b == 0 and hid_dy_b == 0:
                        break
                    await self._send_hid(hid_dx_b, hid_dy_b)
                    await asyncio.sleep(0.03)
                    inner = await self._session.read_cursor_pct()
                    if inner is None:
                        break
                    cx, cy = inner
                    dx_pct = target_aim[0] - cx
                    dy_pct = target_aim[1] - cy
                    if abs(dx_pct) < BIG_DELTA and abs(dy_pct) < BIG_DELTA:
                        break
                await asyncio.sleep(0.08)
                continue

            # Refinement: ask the pointer-accel model for the HID that
            # would land the cursor at the target in one shot.
            if self._pointer_accel is None:
                # No model — fall back to a per-axis open-loop ratio
                # using the established macOS coefficient. Slower
                # convergence but always terminates.
                hid_dx = max(-127, min(127, int(dx_pct / 3e-4)))
                hid_dy = max(-127, min(127, int(dy_pct / 3e-4)))
            else:
                hid_dx, hid_dy = self._pointer_accel.inverse(
                    dx_pct, dy_pct, cx, cy,
                )
                sx, sy = self._pointer_accel_scale()
                if sx != 1.0 or sy != 1.0:
                    hid_dx = int(round(hid_dx * sx))
                    hid_dy = int(round(hid_dy * sy))
            await self._send_hid(int(hid_dx), int(hid_dy))
            await asyncio.sleep(0.10)
            last_pos = (cx, cy)
        else:
            final = await self._session.read_cursor_pct()
            cx, cy = final if final is not None else (last_pos or (0.0, 0.0))
            print(
                f"  Max iters; cursor at ({cx:.2%}, {cy:.2%}); "
                f"residual ({abs(target_aim[0]-cx):.3%}, "
                f"{abs(target_aim[1]-cy):.3%})"
            )
            return ClickOutcome(
                clicked=False, steps=MAX_ITERS,
                reason="max_iters_oracle",
                proof_path=None, history=history,
            )

        if click:
            try:
                await self._session._executor._mouse.click(button)
            except Exception as e:
                logger.warning("click dispatch failed: %s", e)
                return ClickOutcome(
                    clicked=False, steps=it,
                    reason=f"click_dispatch_failed: {e}",
                    proof_path=None, history=history,
                )

        return ClickOutcome(
            clicked=click, steps=it,
            reason="converged_oracle", proof_path=None, history=history,
        )

    async def home_to_pixel(
        self,
        x_pct: float,
        y_pct: float,
        button: str = "left",
        *,
        hotspot_offset: bool = True,
        click: bool = True,
        prev_cursor_pct: tuple[float, float] | None = None,
        dragging: bool = False,
        click_count: int = 1,
    ) -> ClickOutcome:
        """Home the cursor to a pre-located pixel on the webcam frame.

        Skips the OCR/VLM target-location step (operator already told
        us where to click) and skips the post-click LLM navigation
        oracle (manual mode doesn't have a target description to
        verify against). Everything else — slam, cursor detect, visual
        servo, geometric click gate — is the same as ``run()``.

        ``prev_cursor_pct`` is a "no-slam" mode for chained clicks
        within a transient UI (open menu, modal dialog, dropdown):
        the slam-to-corner that normally initialises detection would
        dismiss the open UI on most platforms (LibreOffice closes
        menus the moment the pointer leaves them), so the second
        click in a "File → Exit" sequence would land on whatever's
        underneath the now-vanished menu. When ``prev_cursor_pct`` is
        given, we trust it as the current cursor location, skip the
        slam + oscillation entirely, and just run a motion-diff
        cross-check (small nudge) to refine. If the cross-check
        fails we keep going with the cached position + nudge
        displacement — strictly better than re-slamming, which would
        kill the UI we're trying to navigate.
        """
        session_out = getattr(self._session, "output_dir", None)
        ts = datetime.now().strftime("%H%M%S_manual")
        if session_out is not None:
            run_dir = session_out / "homer" / ts
        else:
            run_dir = PROOF_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        history: list[StepRecord] = []

        # Fast path: when the session has a direct cursor oracle
        # (Quartz on macOS self-capture), the entire HSV / variance /
        # frame-diff cascade is the wrong tool — the cursor isn't in
        # the captured frame to begin with. Run the model-driven closed
        # loop against the oracle instead, mirroring the canary.
        if getattr(self._session, "cursor_reader", None) is not None:
            return await self._home_to_pixel_via_oracle(
                x_pct=x_pct, y_pct=y_pct, button=button,
                hotspot_offset=hotspot_offset, click=click,
                run_dir=run_dir,
            )

        print(
            f"  Manual click homing → ({x_pct:.2%}, {y_pct:.2%}) "
            f"button={button}  no-slam={prev_cursor_pct is not None}"
        )
        print(f"  Step log: {run_dir}/")

        cursor_img: tuple[float, float] | None = None
        if prev_cursor_pct is not None:
            # No-slam path: trust the cached cursor position. We
            # deliberately skip the motion-diff verification nudge
            # here — that nudge would move the cursor 60 HID right,
            # which on a menubar starting position (where most no-
            # slam follow-ups happen) drifts the cursor out of the
            # current column and onto an adjacent menubar item.
            # Adjacent items auto-open via hover-trigger as soon as
            # any sibling menu is already open, so the verification
            # cure would be worse than the disease: a stray
            # diagnostic move ends up dismissing the menu we're
            # trying to navigate. Since the cache is set ONLY after
            # a click_at successfully landed, the cursor is provably
            # at that pixel; no verification is needed.
            cursor_img = prev_cursor_pct
            self._hsv_enabled = True
            logger.info(
                "no-slam: using cached cursor (%.2f%%,%.2f%%)",
                cursor_img[0] * 100, cursor_img[1] * 100,
            )

        if cursor_img is None:
            await self._slam_to_corner()
            await self._send_hid(40, 40)
            await asyncio.sleep(0.30)

            f0 = await self._capture_color()
            cursor_hit = find_cursor_hsv(f0)
            if cursor_hit is not None:
                verified = await self._verify_hsv_by_motion(
                    (cursor_hit.x_pct, cursor_hit.y_pct), run_dir,
                )
                if verified is not None:
                    cursor_img = verified
                    self._hsv_enabled = True
                    print(
                        f"  Cursor (HSV verified): ({cursor_img[0]:.2%}, "
                        f"{cursor_img[1]:.2%})"
                    )
                else:
                    print(
                        "  HSV candidate failed motion verify — falling back."
                    )

        if cursor_img is None:
            print("  Using oscillation-variance to find cursor.")
            osc_hit = await self._find_cursor_via_oscillation(run_dir)
            if osc_hit is None:
                print("  Could not detect cursor by oscillation.")
                return ClickOutcome(
                    clicked=False, steps=0, reason="cursor_not_found",
                    proof_path=None, history=history,
                )
            cursor_img = osc_hit
            print(
                f"  Cursor (oscillation): ({cursor_img[0]:.2%}, "
                f"{cursor_img[1]:.2%})"
            )
            # Post-hoc HSV cross-check: oscillation just gave us the
            # cursor's true position (variance-based, immune to static
            # red UI). If find_cursor_hsv agrees, we know HSV is
            # locking onto the real cursor (not the Reddit logo) and
            # it's safe to use HSV for the cleaner pixel-accurate
            # per-step measurements during the servo loop. Skips the
            # motion-verification dance entirely, which the original
            # gate used and which was rejecting valid candidates
            # whenever pointer-accel or webcam perspective made the
            # observed delta differ from the predicted one.
            try:
                # Differential red-mask cross-check: send a small
                # verification nudge and find the red blob that
                # BECAME red between pre/post frames. Static red UI
                # (icons, syntax highlights, brand accents) cancels
                # out; only the cursor's new position survives the
                # subtraction. Strictly more robust than the single-
                # frame position-aware finder when the screen has any
                # red clutter.
                pre_check = await self._capture_color()
                # 80 HID ≈ 6% image ≈ 128 px → comfortably bigger
                # than the cursor's own footprint (~30 px wide), so
                # pre and post cursor positions don't overlap and
                # the dilated-pre subtraction leaves a clean
                # "newly red" blob. 20 HID isn't enough — the
                # cursor in pre and post overlaps by more than the
                # 4-pixel dilate buffer, eating most of the signal.
                pre_pct = self._pct_per_hid_x or DEFAULT_PCT_PER_HID
                nudge_hid = 80
                await self._send_hid(nudge_hid, 0)
                await asyncio.sleep(SETTLE_SEC)
                post_check = await self._capture_color()
                hsv_check_hit = find_cursor_hsv_motion_directed(
                    pre_check, post_check,
                    cursor_pre_pct=cursor_img,
                    expected_motion_pct=(nudge_hid * pre_pct, 0.0),
                    max_dist_pct=0.12,
                )
                if hsv_check_hit is None:
                    logger.info(
                        "HSV cross-check: no motion-diff red blob "
                        "near osc result; staying on frame-diff."
                    )
                else:
                    self._hsv_enabled = True
                    cursor_img = (
                        hsv_check_hit.x_pct, hsv_check_hit.y_pct,
                    )
                    logger.info(
                        "HSV cross-check OK (motion-diff) — enabling "
                        f"HSV for per-step measurement; adopted "
                        f"({cursor_img[0]:.2%},{cursor_img[1]:.2%})."
                    )
            except Exception as e:
                logger.warning("HSV cross-check error: %s", e)

        # Stash a cursor-template patch for per-step matching in the
        # servo loop, mirroring the equivalent capture point in
        # _run_inner. We need a frame to crop from; f0 was rebound
        # inside the oscillation branch, fall back to a fresh capture
        # if neither branch left one in scope.
        try:
            frame_for_template = f0  # type: ignore[has-type]
        except NameError:
            frame_for_template = await self._capture_color()

        target_img = (float(x_pct), float(y_pct))
        if hotspot_offset:
            hx, hy = self._hotspot_offset_pct()
            target_aim = (
                target_img[0] + hx,
                target_img[1] + hy,
            )
        else:
            target_aim = target_img

        # Long-jump phase: when the slam-to-target distance is large
        # (>15% of image, where the per-step pointer_accel model is
        # out of training distribution), use the long-jump model to
        # predict the full HID needed and fire it as a chain of
        # back-to-back bursts without per-step captures. The standard
        # closed-loop servo below then handles whatever residual
        # remains, in 1-2 iterations instead of 7-10.
        # Skip long-jump in no-slam mode: when the operator is
        # chaining clicks within an open menu/dialog, a multi-burst
        # chain would sweep the cursor diagonally across whatever's
        # next to the current UI element (menubar siblings, dialog
        # neighbours), triggering hover-open on them. Stick with the
        # closed-loop servo's smaller, controlled moves so navigation
        # stays inside the current UI element.
        if self._longjump is not None and prev_cursor_pct is None:
            cursor_img = await self._fire_longjump(
                cursor_img=cursor_img, target_aim=target_aim,
                target_img=target_img, run_dir=run_dir, history=history,
            )

        # Manual clicks demand higher accuracy than the controller's
        # "click on this OCR'd word" flow: an operator picks an exact
        # pixel and expects the cursor to land *there*, not 20px off.
        # Cursor-detection precision (~5–8 px on a 1080p webcam) is
        # the real floor; we set the gate just above that.
        # In no-slam mode (chained clicks within an open UI) we
        # also constrain the closed loop to axis-aligned moves so
        # diagonal transit doesn't open sibling menus.
        return await self._servo_loop(
            target_aim=target_aim, target_img=target_img,
            cursor_img=cursor_img, button=button, run_dir=run_dir,
            history=history, target_desc="<manual>",
            verify_navigation=False, last_proof=None,
            confirm_frames=1,
            # 0.010 (1%) — matches the legacy default. Tightened
            # briefly to 0.006 but the ROI servo can fail to
            # converge on busy backgrounds (page animations being
            # caught by the frame-diff detector as spurious cursor
            # blobs) at that tolerance. 1% is loose enough to
            # converge reliably; final-step measurement noise
            # contributes ±0.5% on top, so worst-case visible
            # click error is ~1.5% (~29 px ≈ 0.4" on 27" 1080p).
            click_tol_pct=0.010,
            click=click,
            axis_aligned=(prev_cursor_pct is not None),
            dragging=dragging,
            click_count=click_count,
        )

    async def drag_to_pixels(
        self,
        from_x_pct: float, from_y_pct: float,
        to_x_pct: float,   to_y_pct: float,
        button: str = "left",
    ) -> ClickOutcome:
        """Press at (from_x_pct, from_y_pct), drag to (to_x_pct, to_y_pct),
        release.

        Composes two ``home_to_pixel(click=False)`` calls with a
        button-press in between. The second home runs in "no-slam"
        mode — it MUST NOT re-slam the cursor to a corner because that
        would translate to a drag across the whole screen with the
        button held, which is destructive in most apps.

        Returns the ClickOutcome from the second home so callers see
        whether the drop landed on target.
        """
        # 1. Home to the source pixel (no click — just position).
        out1 = await self.home_to_pixel(
            from_x_pct, from_y_pct, button=button,
            hotspot_offset=True, click=False,
        )
        if not out1.clicked:
            return out1  # homing failed; nothing to drag
        # 2. Press at the source.
        await self._session._executor._mouse.press(button)
        # 3. Home to the destination with no-slam (prev_cursor_pct
        #    tells home_to_pixel we already know roughly where the
        #    cursor is, so it skips the corner-slam that would
        #    otherwise release-on-no-target across the whole screen).
        try:
            out2 = await self.home_to_pixel(
                to_x_pct, to_y_pct, button=button,
                hotspot_offset=True, click=False,
                prev_cursor_pct=(from_x_pct, from_y_pct),
                dragging=True,
            )
        finally:
            # 4. Always release — leaving the button stuck down is
            #    a worse outcome than an inaccurate drop.
            try:
                await self._session._executor._mouse.release(button)
            except Exception:
                logger.exception("drag release failed")
        return out2

    async def _fire_longjump(
        self, *,
        cursor_img: tuple[float, float],
        target_aim: tuple[float, float],
        target_img: tuple[float, float],
        run_dir: Path,
        history: list[StepRecord],
        min_trigger_pct: float = 0.15,
        max_per_burst: int = 127,
    ) -> tuple[float, float]:
        """Fire a chain of back-to-back HID bursts predicted by the
        long-jump model, without per-step captures. Returns the
        updated cursor position (HSV-tracked if available, otherwise
        the long-jump's open-loop prediction).

        No-op (returns ``cursor_img`` unchanged) when:
          - The long-jump model isn't loaded.
          - The slam-to-target distance is below ``min_trigger_pct``
            (the closed-loop servo is fine for small moves and the
            model's training data was mostly large moves).
        """
        from handsneyes.core.vision.longjump import chunk_hid_for_bursts
        residual = math.hypot(
            target_aim[0] - cursor_img[0],
            target_aim[1] - cursor_img[1],
        )
        if residual < min_trigger_pct or self._longjump is None:
            return cursor_img
        try:
            total_dx, total_dy = self._longjump.predict_total_hid(
                cursor_x_pct=cursor_img[0],
                cursor_y_pct=cursor_img[1],
                target_x_pct=target_aim[0],
                target_y_pct=target_aim[1],
                calibration=getattr(
                    self._longjump, "_calibration", (1.0, 1.0),
                ),
            )
        except Exception as e:
            logger.warning("longjump prediction failed: %s", e)
            return cursor_img
        bursts = chunk_hid_for_bursts(
            total_dx, total_dy, max_per_burst=max_per_burst,
        )
        if not bursts:
            return cursor_img
        logger.info(
            "Long-jump: residual %.1f%% → total_hid=(%d,%d) in %d bursts",
            residual * 100, total_dx, total_dy, len(bursts),
        )
        # Fire all bursts back-to-back with tiny gaps. NO captures
        # between — the whole point is to skip the slow capture loop.
        for hdx, hdy in bursts:
            await self._send_hid(hdx, hdy)
            await asyncio.sleep(0.04)
        # One capture at the end to update the cursor estimate. If
        # HSV is engaged, re-localise near the predicted landing
        # point; else fall back to open-loop prediction.
        predicted = (
            cursor_img[0] + total_dx * self._pct_per_hid_x,
            cursor_img[1] + total_dy * self._pct_per_hid_y,
        )
        await asyncio.sleep(SETTLE_SEC)
        new_pos: tuple[float, float] = predicted
        cursor_found_visually = False
        if self._hsv_enabled:
            try:
                post = await self._capture_color()
                # Progressively widen the HSV search around the
                # predicted landing point. The model's per-axis
                # error is ~3% on val, but tail cases can be >10%,
                # so a fixed-size ring would miss them and we'd
                # blindly pass the open-loop prediction to the
                # closed-loop servo — which then thrashes because
                # its starting cursor estimate is wrong.
                for radius in (0.05, 0.12, 0.25):
                    hit = find_cursor_hsv_near(
                        post, near_pct=predicted, max_dist_pct=radius,
                    )
                    if hit is not None:
                        new_pos = (hit.x_pct, hit.y_pct)
                        cursor_found_visually = True
                        logger.debug(
                            "longjump post-HSV: found at "
                            "(%.2f%%,%.2f%%) within %.0f%% radius",
                            hit.x_pct * 100, hit.y_pct * 100,
                            radius * 100,
                        )
                        break
            except Exception as e:
                logger.debug("longjump post-capture failed: %s", e)
        if not cursor_found_visually:
            # HSV missed altogether — the cursor landed somewhere
            # we didn't expect (or the post-burst frame has motion
            # blur). Fall back to oscillation-variance to get a
            # ground-truth position. Costs ~1.5s but recovers us
            # from a bad open-loop prediction; without it the
            # closed-loop servo would thrash for many iterations
            # before re-discovering the cursor.
            logger.info(
                "longjump: HSV miss after chain — re-localizing "
                "via oscillation"
            )
            osc = await self._find_cursor_via_oscillation(
                run_dir, label="post_longjump",
            )
            if osc is not None:
                new_pos = osc
        _record_step(run_dir, history, StepRecord(
            cursor_img=new_pos, target_img=target_img,
            residual_pct=math.hypot(
                target_aim[0] - new_pos[0],
                target_aim[1] - new_pos[1],
            ),
            hid_dx=total_dx, hid_dy=total_dy,
            ratio_x=self._pct_per_hid_x,
            ratio_y=self._pct_per_hid_y,
            note="longjump_chain",
        ), platform=self._platform_name)
        logger.info(
            "Long-jump landed: cursor=(%.2f%%,%.2f%%) residual=%.2f%% "
            "(predicted=(%.2f%%,%.2f%%))",
            new_pos[0] * 100, new_pos[1] * 100,
            math.hypot(target_aim[0] - new_pos[0],
                       target_aim[1] - new_pos[1]) * 100,
            predicted[0] * 100, predicted[1] * 100,
        )
        return new_pos

    # ────────────────────── target localization ──────────────────────

    async def _locate_target(
        self, image_color: np.ndarray, target_desc: str,
        run_dir: Path,
    ) -> tuple[float, float] | None:
        """Find the target via scene-map + ShowUI grounding.

        Cached for the run — the camera is fixed, so the target's
        image position does not move.
        """
        b64 = await self._encode(image_color)

        # 1) OCR first — if the target is named text (subreddit, link
        # label) and visible on screen, OCR gives us a much more
        # accurate bbox than ShowUI can.
        if have_ocr():
            # When the user gave a quoted target, prioritise it
            # exclusively — generic descriptors like "subreddit",
            # "menu", "entry" are context, not identity, and would
            # otherwise match the first generic occurrence on the page.
            import re as _re
            quoted = _re.findall(r"['\"]([^'\"]+)['\"]", target_desc)
            if quoted:
                primary_keywords = [q.lower() for q in quoted]
            else:
                primary_keywords = ClosedLoopHomer._target_keywords(target_desc)
            print(f"    OCR primary search for keywords {primary_keywords}")
            hits = ocr_find_text(image_color, primary_keywords)
            if hits:
                top = hits[0]
                print(
                    f"    OCR primary matched {top.text!r} at "
                    f"({top.x_pct:.2%},{top.y_pct:.2%}) "
                    f"conf={top.confidence:.0f}"
                )
                try:
                    cv2.imwrite(
                        str(run_dir / "ocr_hit.png"),
                        annotate_ocr_hit(image_color, top),
                    )
                except Exception:
                    pass
                return (top.x_pct, top.y_pct)

        # 2) Scene-map + ShowUI grounding.
        scene = await self._helper._scene_map(b64, run_dir)
        match = self._helper._best_scene_match(scene, target_desc)
        if match is not None:
            print(
                f"  Scene-map matched {match['label']!r} "
                f"({match['description'][:60]}, region={match['region']})"
            )
            label = match["label"]
            stripped = label.lstrip("/").strip()
            for prefix in ("r/", "/r/", "u/"):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            ground_prompts = [
                f"Click on {label}",
                f"Click on the {label} link",
                f"Click on the {label} button",
                f"Click on {stripped}",
                f"Click on the {stripped} link",
                f"Click on the {stripped} subreddit",
                f"Click on r/{stripped}",
                f"Click on the {stripped.lower()} link",
            ]
            seen: set[str] = set()
            for p in ground_prompts:
                key = p.lower()
                if key in seen:
                    continue
                seen.add(key)
                pos = await self._session._showui_query(b64, p)
                if pos is not None:
                    print(f"    ShowUI grounded via {p!r} → {pos}")
                    return pos

        # 3) Fallback: ShowUI directly on the user's description.
        import re as _re
        extra: list[str] = []
        for q in _re.findall(r"['\"]([^'\"]+)['\"]", target_desc):
            extra.extend([
                f"Click on {q}",
                f"Click on the {q} link",
                f"Click on the {q} subreddit",
                f"Click on r/{q}",
                f"Click on r/{q.lower()}",
                f"Click on {q.lower()}",
            ])
        base_prompts = ClosedLoopHomer._showui_prompt_variants(target_desc)
        seen2: set[str] = set()
        for p in (extra + base_prompts):
            key = p.lower()
            if key in seen2:
                continue
            seen2.add(key)
            pos = await self._session._showui_query(b64, p)
            if pos is not None:
                print(f"    ShowUI grounded via fallback {p!r} → {pos}")
                return pos

        # 4) ShowUI on focused crops — when the target is in a small
        # region of the page (sidebar, footer), a crop gives ShowUI a
        # much better chance than the full image.
        crop_regions = [
            ("sidebar_full", 0.0, 0.0, 0.30, 1.0),
            ("sidebar_bottom", 0.0, 0.55, 0.32, 1.0),
            ("footer_strip", 0.0, 0.75, 1.0, 1.0),
        ]
        import re as _re
        quoted_for_crop = _re.findall(r"['\"]([^'\"]+)['\"]", target_desc)
        target_token = (quoted_for_crop[0]
                        if quoted_for_crop else target_desc.split()[-1])
        for name, x0f, y0f, x1f, y1f in crop_regions:
            ih, iw = image_color.shape[:2]
            x0, y0 = int(x0f * iw), int(y0f * ih)
            x1, y1 = int(x1f * iw), int(y1f * ih)
            crop = image_color[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_b64 = await self._encode(crop)
            crop_prompts = [
                f"Click on {target_token}",
                f"Click on the {target_token} link",
                f"Click on r/{target_token}",
            ]
            for cp in crop_prompts:
                pos = await self._session._showui_query(crop_b64, cp)
                if pos is not None:
                    # Map crop fractions back to whole image fractions.
                    crop_w = x1 - x0
                    crop_h = y1 - y0
                    full_x = (pos[0] * crop_w + x0) / iw
                    full_y = (pos[1] * crop_h + y0) / ih
                    print(
                        f"    ShowUI on crop {name!r} grounded {cp!r} "
                        f"→ image=({full_x:.2%},{full_y:.2%})"
                    )
                    return (full_x, full_y)

            # OCR on the same crop with maximum scale, both polarities.
            if have_ocr():
                hits_crop = ocr_find_text(
                    crop, [target_token.lower()],
                    crops=[(0.0, 0.0, 1.0, 1.0)],
                )
                if hits_crop:
                    top = hits_crop[0]
                    crop_w = x1 - x0
                    crop_h = y1 - y0
                    full_x = (top.x_pct * crop_w + x0) / iw
                    full_y = (top.y_pct * crop_h + y0) / ih
                    print(
                        f"    OCR on crop {name!r} matched {top.text!r} "
                        f"→ image=({full_x:.2%},{full_y:.2%})"
                    )
                    return (full_x, full_y)

        # 5) Diagnostic — dump OCR text so we can see what was readable.
        if have_ocr():
            try:
                import pytesseract  # type: ignore
                from handsneyes.core.vision.ocr_finder import (
                    _preprocess_for_ocr,
                )
                full_normal = pytesseract.image_to_string(image_color)
                inv = _preprocess_for_ocr(image_color, scale=4, invert=True)
                full_inv = pytesseract.image_to_string(inv)
                (run_dir / "ocr_full_text.txt").write_text(full_normal)
                (run_dir / "ocr_full_text_inverted.txt").write_text(full_inv)
                cv2.imwrite(
                    str(run_dir / "ocr_inverted_preprocessed.png"), inv,
                )
                print("    OCR + ShowUI all failed — diagnostic dumps saved.")
            except Exception:
                pass
        return None

    # ────────────────────── calibration ──────────────────────

    async def _calibrate(
        self, ref_gray: np.ndarray, run_dir: Path,
    ) -> tuple[tuple[float, float], float, float] | None:
        """Calibrate cursor position and per-axis HID-to-image ratios.

        Sends ``+CALIB_BURST_HID`` along X, then along Y, capturing the
        cursor's movement via frame-diff against the prior frame.
        Returns ``(cursor_img, ratio_x, ratio_y)`` or ``None`` if
        calibration could not detect the cursor.
        """
        # Cursor sits in the corner now (post-slam). Establish baseline.
        await asyncio.sleep(0.10)
        baseline = await self._capture_gray()

        # X burst.
        await self._send_hid(CALIB_BURST_HID, 0)
        await asyncio.sleep(SETTLE_SEC + 0.10)
        post_x = await self._capture_gray()

        x_blobs = self._diff_blobs(baseline, post_x)
        if run_dir is not None:
            self._save_diff_debug(
                run_dir, "calib_x", baseline, post_x, x_blobs,
            )
        # The cursor at "baseline" was near the top-left corner; after
        # +X burst it has moved right. The diff has TWO regions: old
        # cursor position (now-empty) and new cursor position. Pick the
        # one farthest right as the new cursor, leftmost as old cursor.
        if len(x_blobs) < 1:
            return None
        x_blobs.sort(key=lambda b: b["cx"])
        old_x = x_blobs[0]["cx"]
        new_x = x_blobs[-1]["cx"]
        if new_x - old_x < 0.04:
            # Two blobs too close to be reliable cursor motion.
            logger.warning(
                "X-calibration: blobs too close (old=%.2f new=%.2f)",
                old_x, new_x,
            )
            return None
        # Approximate cursor Y in baseline frame as the average of the
        # two blobs' Y (they should both be at the cursor's row in the
        # corner).
        cursor_y_base = (x_blobs[0]["cy"] + x_blobs[-1]["cy"]) / 2

        ratio_x = (new_x - old_x) / CALIB_BURST_HID

        # Y burst from the post-X position.
        await self._send_hid(0, CALIB_BURST_HID)
        await asyncio.sleep(SETTLE_SEC + 0.10)
        post_y = await self._capture_gray()

        y_blobs = self._diff_blobs(post_x, post_y)
        if run_dir is not None:
            self._save_diff_debug(
                run_dir, "calib_y", post_x, post_y, y_blobs,
            )
        if len(y_blobs) < 1:
            return None
        y_blobs.sort(key=lambda b: b["cy"])
        old_y = y_blobs[0]["cy"]
        new_y = y_blobs[-1]["cy"]
        if new_y - old_y < 0.04:
            logger.warning(
                "Y-calibration: blobs too close (old=%.2f new=%.2f)",
                old_y, new_y,
            )
            return None
        cursor_x_after = (y_blobs[0]["cx"] + y_blobs[-1]["cx"]) / 2

        ratio_y = (new_y - old_y) / CALIB_BURST_HID

        # Cursor's image position right now (after both bursts).
        cursor_img = (cursor_x_after, new_y)
        return cursor_img, ratio_x, ratio_y


    def _detect_cursor_in_roi(
        self,
        pre: np.ndarray, post: np.ndarray,
        roi_pct: tuple[float, float, float, float],
        diff_thresh: int = 12,
        near_pct: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float], int] | None:
        """Find the cursor by frame-diffing PRE and POST within ROI.

        ``roi_pct`` is (x, y, w, h) in image-percent. Returns
        ((cx_pct, cy_pct), area_px) or None if no plausible blob.

        The cursor is by construction the only thing that moved
        between PRE and POST when callers diff frames captured
        across a single HID move. Restricting the diff to the ROI
        suppresses everything outside it (page repaint, decorative
        animation, encoder banding far from the cursor's expected
        path).
        """
        if pre.shape != post.shape:
            return None
        if pre.ndim == 3:
            pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        if post.ndim == 3:
            post = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
        h, w = pre.shape[:2]
        # Clamp ROI to frame bounds.
        rx = max(0, int(roi_pct[0] * w))
        ry = max(0, int(roi_pct[1] * h))
        rw = min(w - rx, int(roi_pct[2] * w))
        rh = min(h - ry, int(roi_pct[3] * h))
        if rw <= 0 or rh <= 0:
            return None
        pre_roi = pre[ry:ry+rh, rx:rx+rw]
        post_roi = post[ry:ry+rh, rx:rx+rw]
        diff = cv2.absdiff(pre_roi, post_roi)
        _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        # Cursor diff has TWO blobs: where it WAS (pre position) and
        # where it IS (post position). They'll be similar size. Return
        # the largest (or the one whose movement direction matches the
        # commanded HID — but the caller knows which is "post" by
        # passing pre and post in the right order; we just return the
        # NEW one, which is the one expected at the predicted target).
        candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < 15:  # noise
                continue
            # Aspect-ratio filter against blinking text-input carets
            # WITHOUT also filtering out the macOS I-beam cursor
            # (which appears when the mouse hovers over editable
            # text). Both are tall and thin, but:
            #   - text caret: ~1-2 px wide, single vertical bar
            #   - I-beam cursor: ~4-6 px wide, has serif endpoints
            #     at top and bottom that widen the bounding box
            # So a blob is a caret (and should be rejected) only
            # when it's BOTH very tall-to-wide AND extremely thin
            # in absolute pixels. The I-beam's serifs survive this
            # filter; the caret's vertical line doesn't.
            bx, by, bw, bh = cv2.boundingRect(c)
            if (
                bh > 0 and bw > 0
                and bh / bw > 5.0
                and bw <= 2
                and bh >= 6
            ):
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx_roi = M["m10"] / M["m00"]
            cy_roi = M["m01"] / M["m00"]
            # Convert ROI-relative px back to full-frame pct.
            cx_pct = (rx + cx_roi) / w
            cy_pct = (ry + cy_roi) / h
            candidates.append(((cx_pct, cy_pct), int(a)))
        if not candidates:
            return None
        # Frame-diff returns TWO cursor blobs: the pre-move position
        # and the post-move position. Without disambiguation, picking
        # by area flips between them step-to-step (the cursor moves,
        # the apparent "best" blob alternates). When the caller passes
        # ``near_pct`` (the open-loop predicted post-cursor position),
        # we pick the candidate closest to it — that's the POST blob
        # by construction. Without ``near_pct``, fall back to area
        # (legacy behaviour, kept for non-servo callers).
        if near_pct is not None:
            candidates.sort(
                key=lambda p: math.hypot(
                    p[0][0] - near_pct[0], p[0][1] - near_pct[1],
                ),
            )
        else:
            candidates.sort(key=lambda p: -p[1])
        return candidates[0]

    async def _servo_loop_roi(
        self, *,
        target_aim: tuple[float, float],
        target_img: tuple[float, float],
        cursor_img: tuple[float, float],
        button: str,
        run_dir: Path,
        history: list[StepRecord],
        target_desc: str,
        verify_navigation: bool,
        last_proof: str | None,
        confirm_frames: int = CONFIRM_FRAMES,
        click_tol_pct: float = CLICK_TOL_PCT,
        click: bool = True,
        axis_aligned: bool = False,
        dragging: bool = False,
        click_count: int = 1,
    ) -> ClickOutcome:
        """ROI-stack visual servo (experimental, opt-in via
        HANDSNEYES_ROI_SERVO=1).

        Replaces the proportional-HID inner loop + per-step locator
        cascade with a nested-rectangle approach:
          - Each iteration's detection is restricted to a ROI that
            contains both the aim point and the cursor's last known
            position. Frame-diff suffices — no HSV/template/
            oscillation needed every step.
          - On successful detection, push a tighter ROI (bounding
            box of aim + predicted next cursor position + small
            padding) onto a stack.
          - If detection fails (cursor escaped the ROI), pop to the
            previous larger ROI. If stack empty: fall back to whole-
            frame oscillation re-localize.

        Reuses cruise, slam, and oscillation re-localize verbatim —
        only the inner refinement loop is new.
        """
        # ROI tuning — image-percent units throughout
        INITIAL_PAD = 0.08         # padding around (aim, cursor) for the root ROI
        PUSH_PAD = 0.04            # padding when pushing a tighter ROI
        ROI_MIN_SIDE = 0.06        # don't shrink either dimension below this
        MAX_POP_DEPTH_PER_RUN = 6  # total pops before bailing to oscillation
        MAX_STEPS_ROI = 50
        ROI_SETTLE_SEC = 0.18
        # Cruise threshold: when residual exceeds this, fire a big
        # burst via _send_hid_burst (the same fast traversal the
        # legacy loop uses) and frame-diff in the root ROI to find
        # the cursor at its new position. Below the threshold, fall
        # through to chunked closed-loop refinement.
        ROI_CRUISE_PCT = 0.20
        ROI_CRUISE_SETTLE_S = 0.30

        def _bbox(a: tuple[float, float], b: tuple[float, float],
                  pad: float) -> tuple[float, float, float, float]:
            x0 = max(0.0, min(a[0], b[0]) - pad)
            y0 = max(0.0, min(a[1], b[1]) - pad)
            x1 = min(1.0, max(a[0], b[0]) + pad)
            y1 = min(1.0, max(a[1], b[1]) + pad)
            w = max(ROI_MIN_SIDE, x1 - x0)
            h = max(ROI_MIN_SIDE, y1 - y0)
            # Re-clip after enforcing min side
            x0 = max(0.0, min(1.0 - w, x0))
            y0 = max(0.0, min(1.0 - h, y0))
            return (x0, y0, w, h)

        # ROI stack. Each entry is (roi, cursor_pos_at_push).
        roi_stack: list[
            tuple[tuple[float, float, float, float], tuple[float, float]]
        ] = []
        roi_stack.append((_bbox(target_aim, cursor_img, INITIAL_PAD), cursor_img))
        print(
            f"  [ROI servo] aim=({target_aim[0]:.2%},{target_aim[1]:.2%}) "
            f"start_cursor=({cursor_img[0]:.2%},{cursor_img[1]:.2%}) "
            f"root_roi={roi_stack[0][0]}"
        )

        pops_used = 0
        confirm_count = 0
        # Tracks consecutive root-ROI detection failures (no blob in
        # ROI, escape-pop also failed). Drives the no-slam "trust
        # the cache" fast-bail below.
        consec_relocalize_fails = 0
        best_residual = math.hypot(
            target_aim[0] - cursor_img[0], target_aim[1] - cursor_img[1],
        )

        for step in range(1, MAX_STEPS_ROI + 1):
            roi, _ = roi_stack[-1]
            t_step = time.monotonic()

            dx_pct = target_aim[0] - cursor_img[0]
            dy_pct = target_aim[1] - cursor_img[1]
            residual = math.hypot(dx_pct, dy_pct)
            if residual < best_residual:
                best_residual = residual

            if residual <= click_tol_pct:
                confirm_count += 1
                print(
                    f"  [{step:02d}|R] cursor=({cursor_img[0]:.2%},"
                    f"{cursor_img[1]:.2%}) residual={residual:.2%} "
                    f"— confirm {confirm_count}/{confirm_frames}"
                )
                if confirm_count >= confirm_frames:
                    # Post-confirm verification: the previous step's
                    # measurement said residual ≤ tol, but single-
                    # frame frame-diff jitters ±0.5% — the true
                    # cursor position could be up to ~0.5% further
                    # off. Re-measure via a tiny wiggle (~15 HID,
                    # undone) for a fresh independent reading
                    # before clicking. If the fresh reading puts
                    # residual back above tol, the previous reading
                    # was a noisy under-estimate; resume servoing
                    # rather than click off-target.
                    verified = await self._verify_cursor_via_wiggle(cursor_img)
                    if verified is not None:
                        v_resid = math.hypot(
                            target_aim[0] - verified[0],
                            target_aim[1] - verified[1],
                        )
                        if v_resid > click_tol_pct:
                            cursor_img = verified
                            confirm_count = 0
                            print(
                                f"  [{step:02d}|V] post-confirm "
                                f"re-measure: cursor=({verified[0]:.2%},"
                                f"{verified[1]:.2%}) residual={v_resid:.2%}"
                                f" > {click_tol_pct:.1%} — resuming"
                            )
                            continue
                        cursor_img = verified

                        # ─── Micro-correction loop ───────────────
                        # The tolerance gate accepts residual ≤ 1%,
                        # but on a 27" 1080p display 1% is still ~5
                        # mm visual error. After the wiggle confirms
                        # we're sub-tol, do up to N tiny corrective
                        # nudges (each followed by another wiggle-
                        # measure) to drive residual down toward
                        # the camera's measurement noise floor. Bail
                        # if any measurement fails OR if a step
                        # makes residual worse (HID undershot /
                        # overshot due to pointer-accel nonlinearity
                        # at small magnitudes).
                        MICRO_PRECISION_TARGET = 0.003  # ≈ 0.6 mm
                        MICRO_MAX_ITERS = 4
                        prev_resid = v_resid
                        for micro in range(MICRO_MAX_ITERS):
                            if v_resid <= MICRO_PRECISION_TARGET:
                                break
                            dx_micro = target_aim[0] - cursor_img[0]
                            dy_micro = target_aim[1] - cursor_img[1]
                            mhid_dx, mhid_dy = self._hid_for_residual(
                                dx_micro, dy_micro,
                            )
                            # Clamp to small nudges — at this point
                            # we're chasing pixels, not percent.
                            mhid_dx = max(-15, min(15, mhid_dx))
                            mhid_dy = max(-15, min(15, mhid_dy))
                            if mhid_dx == 0 and mhid_dy == 0:
                                break
                            await self._send_hid(mhid_dx, mhid_dy)
                            await asyncio.sleep(0.10)
                            re_verified = await self._verify_cursor_via_wiggle(
                                cursor_img,
                            )
                            if re_verified is None:
                                break
                            new_resid = math.hypot(
                                target_aim[0] - re_verified[0],
                                target_aim[1] - re_verified[1],
                            )
                            # Made things worse — back out the move
                            # (cursor is in a non-linear accel zone)
                            # and stop micro-correcting; we'll click
                            # at the better prior position.
                            if new_resid > prev_resid + 0.001:
                                await self._send_hid(-mhid_dx, -mhid_dy)
                                await asyncio.sleep(0.08)
                                print(
                                    f"  [{step:02d}|µ{micro+1}] hid=({mhid_dx:+d},"
                                    f"{mhid_dy:+d}) → resid={new_resid:.2%} > "
                                    f"prior {prev_resid:.2%}; reverted, "
                                    f"clicking at prior position"
                                )
                                break
                            cursor_img = re_verified
                            v_resid = new_resid
                            prev_resid = new_resid
                            print(
                                f"  [{step:02d}|µ{micro+1}] hid=({mhid_dx:+d},"
                                f"{mhid_dy:+d}) → cursor=({re_verified[0]:.2%},"
                                f"{re_verified[1]:.2%}) residual={new_resid:.2%}"
                            )

                        # ─── DINOv2 click-target snap ──────────
                        # Opt-in (HANDSNEYES_DINO_SNAP=1). Looks at
                        # a 224×224 ROI around the aim, finds the
                        # coherent UI element under the click, and
                        # snaps the cursor to its centroid if it's
                        # within the snap radius. Adds ~150 ms of
                        # inference but can rescue clicks that
                        # geometrically landed a few px off a button.
                        if (
                            os.environ.get("HANDSNEYES_DINO_SNAP") == "1"
                        ):
                            try:
                                from handsneyes.core.vision.dino_snap import (
                                    find_snap_target,
                                )
                                frame = await self._capture_color()
                                snap = find_snap_target(
                                    frame, target_aim,
                                    snap_radius_pct=0.03,
                                )
                                if snap is not None:
                                    # Drive cursor to the snap point
                                    # via one final micro-correction.
                                    sdx = snap[0] - cursor_img[0]
                                    sdy = snap[1] - cursor_img[1]
                                    if abs(sdx) > 0.001 or abs(sdy) > 0.001:
                                        shx, shy = self._hid_for_residual(sdx, sdy)
                                        shx = max(-20, min(20, shx))
                                        shy = max(-20, min(20, shy))
                                        if shx != 0 or shy != 0:
                                            await self._send_hid(shx, shy)
                                            await asyncio.sleep(0.10)
                                        cursor_img = snap
                                        print(
                                            f"  [{step:02d}|D] dino snap: "
                                            f"target_aim=({target_aim[0]:.2%},"
                                            f"{target_aim[1]:.2%}) → "
                                            f"snap=({snap[0]:.2%},{snap[1]:.2%}) "
                                            f"Δ=({sdx*100:+.2f}%,{sdy*100:+.2f}%) "
                                            f"hid=({shx:+d},{shy:+d})"
                                        )
                                else:
                                    print(
                                        f"  [{step:02d}|D] dino: no snap "
                                        f"target found near aim"
                                    )
                            except Exception as e:
                                print(f"  [{step:02d}|D] dino skipped: {e}")

                        print(
                            f"  [{step:02d}|V] verified: cursor=({cursor_img[0]:.2%},"
                            f"{cursor_img[1]:.2%}) residual={v_resid:.2%}"
                            f" ≤ {click_tol_pct:.1%} — clicking"
                        )
                    # else: wiggle couldn't localise (busy bg) —
                    # fall through to click with prior cursor_img.
                    return await self._roi_commit_click(
                        target_aim=target_aim, target_img=target_img,
                        cursor_img=cursor_img, button=button,
                        click=click, click_count=click_count,
                        run_dir=run_dir, step=step, history=history,
                    )
                continue
            else:
                confirm_count = 0

            # Cruise mode: large residual → fast burst via the
            # unchunked Pi endpoint. macOS sees a single high-velocity
            # event and applies its high-speed accel curve, so a 220-
            # HID burst covers much more ground than 220 chunked HID.
            # Frame-diff in the (large) root ROI still finds the
            # cursor post-burst because the cursor's pre/post blobs
            # are far apart in the diff mask.
            if residual > ROI_CRUISE_PCT and not dragging:
                fr_x = max(
                    RATIO_MIN,
                    min(RATIO_MAX * 10, self._pct_per_hid_fast_x),
                )
                fr_y = max(
                    RATIO_MIN,
                    min(RATIO_MAX * 10, self._pct_per_hid_fast_y),
                )
                hid_dx = 0
                hid_dy = 0
                if abs(dx_pct) >= 1e-4:
                    h = int(abs(dx_pct) / fr_x)
                    h = max(MIN_HID_PER_AXIS, min(MAX_HID_PER_AXIS, h))
                    hid_dx = h if dx_pct > 0 else -h
                if abs(dy_pct) >= 1e-4:
                    h = int(abs(dy_pct) / fr_y)
                    h = max(MIN_HID_PER_AXIS, min(MAX_HID_PER_AXIS, h))
                    hid_dy = h if dy_pct > 0 else -h
                if axis_aligned:
                    if abs(dx_pct) > abs(dy_pct):
                        hid_dy = 0
                    else:
                        hid_dx = 0
                pre = await self._capture_gray()
                await self._send_hid_burst(hid_dx, hid_dy)
                await asyncio.sleep(ROI_CRUISE_SETTLE_S)
                post = await self._capture_gray()
                # Predict post-burst position using FAST ratio so the
                # disambiguator picks the right diff blob.
                predicted = (
                    cursor_img[0] + hid_dx * fr_x,
                    cursor_img[1] + hid_dy * fr_y,
                )
                hit = self._detect_cursor_in_roi(
                    pre, post, roi, near_pct=predicted,
                )
                if hit is not None:
                    pre_cursor_pos = cursor_img
                    cursor_img = hit[0]
                    # Refine fast ratio from observed delta
                    self._refine_fast_ratio(
                        hid_dx, hid_dy,
                        cursor_img[0] - pre_cursor_pos[0],
                        cursor_img[1] - pre_cursor_pos[1],
                    )
                    # Tighten ROI (same as closed-loop branch)
                    new_roi = _bbox(target_aim, cursor_img, PUSH_PAD)
                    cur_area = roi[2] * roi[3]
                    new_area = new_roi[2] * new_roi[3]
                    if new_area < cur_area * 0.85:
                        roi_stack.append((new_roi, cursor_img))
                    elapsed = time.monotonic() - t_step
                    print(
                        f"  [{step:02d}|>] cruise hid=({hid_dx:+4d},"
                        f"{hid_dy:+4d}) cursor→({cursor_img[0]:.2%},"
                        f"{cursor_img[1]:.2%}) resid={residual:.2%} "
                        f"fr=({self._pct_per_hid_fast_x*1000:.3f},"
                        f"{self._pct_per_hid_fast_y*1000:.3f})‰ "
                        f"stack={len(roi_stack)} {elapsed:.2f}s"
                    )
                    continue
                # Cruise burst missed in ROI — fall through to
                # chunked closed-loop refinement below.

            # Axis-aligned mode: zero the smaller-residual axis so the
            # cursor moves in an L-shape (no diagonal sweep across
            # sibling UI elements).
            hid_dx, hid_dy = self._hid_for_residual(dx_pct, dy_pct)
            if axis_aligned:
                if abs(dx_pct) > abs(dy_pct):
                    hid_dy = 0
                else:
                    hid_dx = 0

            pre = await self._capture_gray()
            await self._send_hid(hid_dx, hid_dy)
            await asyncio.sleep(ROI_SETTLE_SEC)
            post = await self._capture_gray()

            # Predict where the cursor SHOULD have landed (open-loop
            # estimate from the homer's current ratio). Used to
            # disambiguate the pre/post blobs from the frame-diff.
            predicted = (
                cursor_img[0] + hid_dx * self._pct_per_hid_x,
                cursor_img[1] + hid_dy * self._pct_per_hid_y,
            )

            hit = self._detect_cursor_in_roi(
                pre, post, roi, near_pct=predicted,
            )
            if hit is None:
                # Cursor escaped this ROI. Pop to the previous larger
                # ROI and retry detection there. Don't push smaller —
                # we lost track once.
                if len(roi_stack) > 1 and pops_used < MAX_POP_DEPTH_PER_RUN:
                    roi_stack.pop()
                    pops_used += 1
                    bigger_roi, restored_cursor = roi_stack[-1]
                    # Re-attempt detection inside the bigger ROI.
                    hit_retry = self._detect_cursor_in_roi(
                        pre, post, bigger_roi, near_pct=predicted,
                    )
                    if hit_retry is not None:
                        cursor_img = hit_retry[0]
                        print(
                            f"  [{step:02d}|↑] cursor escaped — popped to "
                            f"larger ROI; refound at ({cursor_img[0]:.2%},"
                            f"{cursor_img[1]:.2%})"
                        )
                        continue
                    # Still missing — restore cursor_img to the
                    # last-known good position and try smaller hid
                    # next iteration.
                    cursor_img = restored_cursor
                    print(
                        f"  [{step:02d}|↑] cursor escaped — popped, "
                        f"detection failed too; restoring cursor=({cursor_img[0]:.2%},"
                        f"{cursor_img[1]:.2%}) and continuing"
                    )
                    continue
                # Stack at root, or we've popped too many times.
                # Trust-the-cache fast bail: in no-slam mode the
                # cached cursor position is "where the previous
                # click landed", which on a remote screen-share
                # target may be the most reliable signal we have
                # (frame-diff in a tight ROI on a busy/focused UI
                # is noisy — caret blinks, focus highlights, video
                # encoder banding all show up). Rather than grind
                # through oscillation re-localizes that may also
                # fail, after 3 consecutive root-ROI detection
                # failures in no-slam mode, just trust the cached
                # cursor and commit — the click will land within
                # ~1-2 % of intent (the residual we computed at
                # loop start), which is better than burning 30+
                # steps trying to drive it lower via a broken
                # detection signal.
                consec_relocalize_fails += 1
                if (
                    axis_aligned  # = no-slam mode
                    and consec_relocalize_fails >= 3
                    and residual < 0.05
                ):
                    print(
                        f"  [{step:02d}|?] {consec_relocalize_fails} consecutive "
                        f"detection failures in no-slam mode with residual "
                        f"{residual:.2%} — trusting cached cursor and clicking"
                    )
                    return await self._roi_commit_click(
                        target_aim=target_aim, target_img=target_img,
                        cursor_img=cursor_img, button=button,
                        click=click, click_count=click_count,
                        run_dir=run_dir, step=step, history=history,
                    )
                print(
                    f"  [{step:02d}|?] root ROI detection failed "
                    f"(pops={pops_used}) — oscillation re-localize"
                )
                relocated = await self._find_cursor_via_oscillation(
                    run_dir, label=f"step{step:02d}_relocate",
                )
                if relocated is not None:
                    cursor_img = relocated
                    # Rebuild ROI stack from scratch around new cursor.
                    roi_stack = [
                        (_bbox(target_aim, cursor_img, INITIAL_PAD), cursor_img),
                    ]
                    pops_used = 0
                else:
                    print(
                        "  [ROI servo] oscillation re-localize failed — "
                        "best-effort click at last-known cursor"
                    )
                    break
                continue

            new_cursor, blob_area = hit
            pre_cursor_pos = cursor_img
            cursor_img = new_cursor
            consec_relocalize_fails = 0

            # Refine the chunked-mode pct-per-HID ratio from the
            # observed motion. Same EMA update the legacy servo
            # used — without this, _hid_for_residual systematically
            # over-estimates how far the cursor will move per HID
            # unit (on screen-share remote targets the chunked
            # ratio is ~0.15‰, well below the DEFAULT_PCT_PER_HID
            # = 0.833‰ seed), and closed-loop steps shrink by only
            # ~12% per iteration instead of the design 55%. With
            # refinement, residual halves every 1-2 steps once the
            # ratio is calibrated.
            self._refine_ratio(
                hid_dx, hid_dy,
                cursor_img[0] - pre_cursor_pos[0],
                cursor_img[1] - pre_cursor_pos[1],
            )

            # Push a tighter ROI. The new ROI bounds the aim and the
            # predicted next cursor position (where we expect to end
            # up after the next step). Padded so micro-overshoots
            # stay inside.
            new_roi = _bbox(target_aim, cursor_img, PUSH_PAD)
            # Don't push if the new ROI isn't actually tighter than
            # the current one — avoids stack growth on no-progress.
            cur_area = roi[2] * roi[3]
            new_area = new_roi[2] * new_roi[3]
            if new_area < cur_area * 0.85:
                roi_stack.append((new_roi, cursor_img))

            elapsed = time.monotonic() - t_step
            print(
                f"  [{step:02d}|R] hid=({hid_dx:+4d},{hid_dy:+4d}) "
                f"cursor→({cursor_img[0]:.2%},{cursor_img[1]:.2%}) "
                f"resid={residual:.2%} "
                f"roi=({roi[0]:.2%},{roi[1]:.2%},{roi[2]:.2%}×{roi[3]:.2%}) "
                f"stack={len(roi_stack)} blob={blob_area}px {elapsed:.2f}s"
            )

        # Loop exited without geometric_confirm. Best-effort click at
        # whatever cursor position we last saw — matches legacy
        # behaviour.
        print(
            f"  [ROI servo] exhausted budget. best_residual={best_residual:.2%}; "
            f"falling back to best-effort click"
        )
        return await self._roi_commit_click(
            target_aim=target_aim, target_img=target_img,
            cursor_img=cursor_img, button=button,
            click=click, click_count=click_count,
            run_dir=run_dir, step=MAX_STEPS_ROI, history=history,
            best_effort=True,
        )

    async def _verify_cursor_via_wiggle(
        self, last_cursor: tuple[float, float],
        search_pad: float = 0.05,
    ) -> tuple[float, float] | None:
        """Re-measure the cursor's exact position via a small known
        wiggle just before committing the click.

        Why: single-frame frame-diff jitters ±0.5% on a screen-share
        remote target. If the cursor's true residual is 1.4% but a
        noisy measurement reported 0.9%, the tolerance gate fires
        and we click 1.4% off target. Replacing the cursor estimate
        with a fresh wiggle-localised measurement averages that
        single-frame noise out: the wiggle's pre/post pair gives a
        differential signal independent of which frame caught the
        noise spike.

        Returns the cursor's *pre-wiggle* position (= post-undo
        position — the cursor lands back where it started after the
        wiggle is reversed). Returns None if the wiggle's diff
        couldn't isolate the cursor (e.g. heavy background activity).
        """
        WIGGLE_HID = 15
        WIGGLE_SETTLE_S = 0.12
        try:
            pre = await self._capture_gray()
            await self._send_hid(WIGGLE_HID, WIGGLE_HID)
            await asyncio.sleep(WIGGLE_SETTLE_S)
            post = await self._capture_gray()
            # Undo so the cursor returns to its pre-wiggle position
            # — caller must not see any net cursor displacement.
            await self._send_hid(-WIGGLE_HID, -WIGGLE_HID)
        except Exception:
            return None

        # Expected wiggle delta in image-pct, using the homer's
        # current chunked ratio (which has been refined throughout
        # the servo loop, so it's well-calibrated by now).
        expected_dx_pct = WIGGLE_HID * self._pct_per_hid_x
        expected_dy_pct = WIGGLE_HID * self._pct_per_hid_y
        predicted_post = (
            last_cursor[0] + expected_dx_pct,
            last_cursor[1] + expected_dy_pct,
        )

        # Tight ROI around the predicted post-wiggle position —
        # we know the cursor is RIGHT THERE because the wiggle's
        # only ~15 HID = ~2-3 px.
        roi = (
            max(0.0, predicted_post[0] - search_pad),
            max(0.0, predicted_post[1] - search_pad),
            min(2 * search_pad, 1.0),
            min(2 * search_pad, 1.0),
        )
        hit = self._detect_cursor_in_roi(
            pre, post, roi, near_pct=predicted_post,
        )
        if hit is None:
            return None
        post_cursor, _ = hit
        # Convert post-wiggle measurement back to pre-wiggle
        # position (= where the cursor sits NOW, after undo).
        return (
            post_cursor[0] - expected_dx_pct,
            post_cursor[1] - expected_dy_pct,
        )

    def _measure_click_feedback_offset(
        self,
        pre_color: np.ndarray, post_color: np.ndarray,
        target_img: tuple[float, float],
        target_aim: tuple[float, float],
        search_radius_pct: float = 0.07,
    ) -> tuple[float, float] | None:
        """Find where a click visibly registered by frame-diffing
        pre/post within a small ROI around the user's intended
        target point, then return (delta_x_pct, delta_y_pct) from
        the target.

        Visible click effects include: text caret appearance/move,
        button focus ring, link focus underline, app dock bounce,
        menu open, hover-deselection. As long as SOMETHING changes
        within ``search_radius_pct`` of the target, the diff finds
        it. The largest changed region (closest to target if tied)
        is the click feedback.

        The cursor itself sits at ``target_aim`` (= target + current
        hotspot offset) and may show small diff signature due to
        encoder noise — usually smaller than the actual feedback
        signature, so largest-blob picks the right one. When the
        feedback is exactly on target, the returned delta is ~zero
        and the caller leaves the hotspot offset unchanged.
        """
        if pre_color is None or post_color is None:
            return None
        if pre_color.shape != post_color.shape:
            return None
        h, w = pre_color.shape[:2]
        cx = int(target_img[0] * w)
        cy = int(target_img[1] * h)
        rx = int(search_radius_pct * w)
        ry = int(search_radius_pct * h)
        x0 = max(0, cx - rx); y0 = max(0, cy - ry)
        x1 = min(w, cx + rx); y1 = min(h, cy + ry)
        if x1 <= x0 or y1 <= y0:
            return None
        pre_g = cv2.cvtColor(pre_color[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        post_g = cv2.cvtColor(post_color[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(pre_g, post_g)
        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        best = None  # (area, cx_pct, cy_pct)
        for c in contours:
            a = cv2.contourArea(c)
            if a < 20:  # noise
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            bx = (x0 + M["m10"] / M["m00"]) / w
            by = (y0 + M["m01"] / M["m00"]) / h
            if best is None or a > best[0]:
                best = (a, bx, by)
        if best is None:
            return None
        _, fx, fy = best
        return (fx - target_img[0], fy - target_img[1])

    def _update_hotspot_from_feedback(
        self,
        target_img: tuple[float, float],
        target_aim: tuple[float, float],
        feedback_delta: tuple[float, float],
        alpha: float = 0.15,
    ) -> bool:
        """Update the calibrated hotspot offset from observed click-
        feedback delta. Returns True if the update was applied.

        Derivation:
            target_aim = target_img + H_used
            click registers at: target_aim - H_true = target_img +
              (H_used - H_true)
            so feedback_delta = feedback_pos - target_img =
              H_used - H_true
            => H_true = H_used - feedback_delta

        Conservative EMA (alpha=0.15) on top of the slam-corner
        calibration: the click-feedback measurement is noisy
        because the diff can also catch unrelated background
        animations / blink carets / focus loss events. Slam-corner
        gives a tight measurement; click-feedback refines around
        that anchor over many clicks rather than replacing it on
        a single noisy sample.

        Rejects measurements where |delta| > REJECT_DELTA_PCT
        (2.5%) — those are almost certainly false positives. The
        cursor's centroid-to-hotspot distance lives between 0.5%
        and 2% in practice (verified against multiple slam-corner
        calibrations), so a 4-5% delta means the diff caught an
        unrelated UI change rather than the click's visible effect.
        """
        REJECT_DELTA_PCT = 0.025
        if (abs(feedback_delta[0]) > REJECT_DELTA_PCT
                or abs(feedback_delta[1]) > REJECT_DELTA_PCT):
            return False
        H_used = (
            target_aim[0] - target_img[0],
            target_aim[1] - target_img[1],
        )
        # The "ideal" H suggested by THIS observation alone.
        H_obs = (
            H_used[0] - feedback_delta[0],
            H_used[1] - feedback_delta[1],
        )
        # EMA blend toward the observation. With alpha=0.15, a
        # single bad sample shifts the offset by at most ~1% even
        # if its delta was the max-accepted 8%.
        H_blend = (
            alpha * H_obs[0] + (1 - alpha) * H_used[0],
            alpha * H_obs[1] + (1 - alpha) * H_used[1],
        )
        # Sanity clamp: cursor hotspot shouldn't be more than ±10%
        # from its centroid (10% on 1920w = ~190 px, a huge cursor).
        H_blend = (
            max(-0.10, min(0.10, H_blend[0])),
            max(-0.10, min(0.10, H_blend[1])),
        )
        self._calibrated_hotspot_offset = H_blend
        return True

    async def _roi_commit_click(
        self, *, target_aim, target_img, cursor_img, button, click,
        click_count, run_dir, step, history,
        best_effort: bool = False,
    ) -> ClickOutcome:
        """Fire the click + multi-click extras inside the geometric-
        confirm window, capture proof, return. Mirrors the equivalent
        sequence in the legacy _servo_loop's confirm block — shared
        here so the ROI servo doesn't have to reimplement the count>1
        + proof-capture + history-record machinery.
        """
        # Capture pre-click frame for click-feedback calibration.
        pre_click_color: np.ndarray | None = None
        if click:
            try:
                pre_click_color = await self._capture_color()
            except Exception:
                pre_click_color = None
            # Native multi-click: passing count > 1 sequences the
            # clicks on the Pi with ~40 ms inter-click timing
            # (vs ~150 ms when the dev side dispatched each click
            # separately and the per-click HTTP roundtrip stacked
            # up). Well within macOS's double-click window even on
            # "Fast" user settings.
            try:
                await self._session._executor._mouse.click(
                    button, count=click_count,
                )
            except TypeError:
                # Older backend without count= parameter — fall
                # back to dev-side sequencing.
                await self._session._executor._mouse.click(button)
                for _ in range(1, click_count):
                    await asyncio.sleep(0.08)
                    try:
                        await self._session._executor._mouse.click(button)
                    except Exception:
                        break
        post_click_color: np.ndarray | None = None
        try:
            await asyncio.sleep(0.4)
            proof = await self._capture_proof(run_dir, step * 100)
            # _capture_proof persists the frame; grab a fresh capture
            # too for the feedback diff (the proof file format may
            # have been compressed/resized).
            post_click_color = await self._capture_color()
        except Exception:
            proof = None

        # Click-feedback hotspot calibration. Adjusts
        # _calibrated_hotspot_offset based on where the click's
        # visible effect appeared relative to the user's target.
        # Skips silently if no feedback was detected (clicked on
        # inert background).
        if click and pre_click_color is not None and post_click_color is not None:
            delta = self._measure_click_feedback_offset(
                pre_click_color, post_click_color, target_img, target_aim,
            )
            if delta is not None:
                prior = self._calibrated_hotspot_offset
                applied = self._update_hotspot_from_feedback(
                    target_img, target_aim, delta,
                )
                if applied:
                    new = self._calibrated_hotspot_offset
                    prior_str = (
                        f"({prior[0]*100:.2f}%, {prior[1]*100:.2f}%)"
                        if prior is not None else "(default)"
                    )
                    print(
                        f"  [click-feedback] delta=({delta[0]*100:+.2f}%,"
                        f"{delta[1]*100:+.2f}%) hotspot offset: "
                        f"{prior_str} → ({new[0]*100:.2f}%, {new[1]*100:.2f}%)"
                    )
                else:
                    print(
                        f"  [click-feedback] delta=({delta[0]*100:+.2f}%,"
                        f"{delta[1]*100:+.2f}%) — exceeds sanity threshold, "
                        f"likely false positive (unrelated UI change). "
                        f"Hotspot offset unchanged."
                    )
            else:
                print(
                    f"  [click-feedback] no visible change near "
                    f"target — hotspot offset unchanged"
                )
        residual = math.hypot(
            target_aim[0] - cursor_img[0], target_aim[1] - cursor_img[1],
        )
        _record_step(run_dir, history, StepRecord(
            cursor_img=cursor_img, target_img=target_img,
            residual_pct=residual, hid_dx=0, hid_dy=0,
            ratio_x=self._pct_per_hid_x,
            ratio_y=self._pct_per_hid_y,
            note="roi_servo;best_effort" if best_effort else "roi_servo;geometric",
        ), platform=self._platform_name)
        return ClickOutcome(
            clicked=True, steps=step,
            reason="best_effort" if best_effort else "geometric_confirm",
            proof_path=proof, history=history,
        )

    def _hid_for_residual(
        self, dx_pct: float, dy_pct: float,
    ) -> tuple[int, int]:
        """Compute HID move from residual using current (clamped) ratio,
        with a hard ceiling per axis."""
        ax = abs(dx_pct)
        ay = abs(dy_pct)
        ratio_x = max(RATIO_MIN, min(RATIO_MAX, self._pct_per_hid_x))
        ratio_y = max(RATIO_MIN, min(RATIO_MAX, self._pct_per_hid_y))
        hid_x = 0
        hid_y = 0
        if ax >= 1e-4:
            hid_units_x = int(ax / ratio_x * STEP_DISTANCE_FRACTION)
            if hid_units_x < MIN_HID_PER_AXIS:
                hid_units_x = MIN_HID_PER_AXIS if ax > CLICK_TOL_PCT else 0
            if hid_units_x > MAX_HID_PER_AXIS:
                hid_units_x = MAX_HID_PER_AXIS
            hid_x = hid_units_x if dx_pct > 0 else -hid_units_x
        if ay >= 1e-4:
            hid_units_y = int(ay / ratio_y * STEP_DISTANCE_FRACTION)
            if hid_units_y < MIN_HID_PER_AXIS:
                hid_units_y = MIN_HID_PER_AXIS if ay > CLICK_TOL_PCT else 0
            if hid_units_y > MAX_HID_PER_AXIS:
                hid_units_y = MAX_HID_PER_AXIS
            hid_y = hid_units_y if dy_pct > 0 else -hid_units_y
        return hid_x, hid_y

    def _refine_ratio(
        self,
        hid_dx: int, hid_dy: int,
        measured_dx_pct: float, measured_dy_pct: float,
    ) -> None:
        """EMA-update the per-axis ratio using observed motion.

        Only refine when:
        - the move was ≥ MIN_HID_PER_AXIS (otherwise no signal)
        - the observed motion is in the SAME direction as commanded
        - the observed motion magnitude is ≥ 25% of expected
          (otherwise HSV likely mis-located and we'd corrupt the ratio).
        Clamp the result to [RATIO_MIN, RATIO_MAX] so a bad sample can
        never run the loop off the rails.
        """
        for hid, meas, axis in (
            (hid_dx, measured_dx_pct, "x"),
            (hid_dy, measured_dy_pct, "y"),
        ):
            if abs(hid) < MIN_HID_PER_AXIS or meas == 0:
                continue
            if (hid > 0) != (meas > 0):
                continue  # observed direction disagrees → noise
            current = (self._pct_per_hid_x if axis == "x"
                       else self._pct_per_hid_y)
            expected = abs(hid) * current
            # 0.05 instead of 0.25: when the homer is started against a
            # target whose accel curve differs sharply from the default
            # (e.g. cross-mac control through a screen-share virtual
            # camera), the FIRST honest measurement is below the
            # default "noise" threshold and would be rejected, which
            # locks the ratio at the seed forever. 0.05 still rejects
            # genuine HSV mis-detects (true noise sits around 10× below
            # this) but admits the legitimate cross-target signal.
            if abs(meas) < 0.05 * expected:
                continue  # observed too small → likely HSV mis-detect
            obs = meas / hid
            if not (RATIO_MIN <= abs(obs) <= RATIO_MAX):
                continue
            new = RATIO_EMA * obs + (1 - RATIO_EMA) * current
            new = max(RATIO_MIN, min(RATIO_MAX, abs(new)))
            if axis == "x":
                self._pct_per_hid_x = new
            else:
                self._pct_per_hid_y = new

    # ────────────────────── plumbing ──────────────────────


    def _pointer_accel_scale(self) -> tuple[float, float]:
        """UI-controlled multipliers applied to the model's HID output.

        Read from the session's ``ctx.scratch`` (snapshotted there by
        the cc factory at run start from the runtime_state values the
        UI writes via ``/api/pointer-accel-scale``). Default is
        ``(1.0, 1.0)`` — no-op — so legacy code paths and direct CLI
        invocations of the homer are unaffected.

        Used to bridge the dev / target effective-resolution gap
        without retraining: the same HID moves the same physical
        pixels on both machines, but those pixels are a different
        percent of each machine's screen, so the model's
        percent-keyed predictions need a constant rescale per axis.
        """
        ctx = getattr(self._session, "_ctx", None)
        if ctx is None:
            return 1.0, 1.0
        scratch = getattr(ctx, "scratch", None) or {}
        sx = float(scratch.get("pointer_accel_scale_x", 1.0))
        sy = float(scratch.get("pointer_accel_scale_y", 1.0))
        return sx, sy

    async def _slam_to_corner(self) -> None:
        print("  Slamming to top-left corner...")
        for _ in range(200):
            try:
                await self._session._executor._mouse.move(-20, -20)
            except Exception:
                pass
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.3)
        # Auto-calibrate hotspot offset using the corner clamp.
        # See _calibrate_hotspot_at_corner. Skipped if already
        # calibrated this session (cheap idempotency).
        if self._calibrated_hotspot_offset is None:
            await self._calibrate_hotspot_at_corner()

    async def _calibrate_hotspot_at_corner(self) -> None:
        """Auto-calibrate the hotspot offset right after slam-to-
        corner. macOS (and most OSes) clamp the cursor's HOTSPOT at
        the screen edge, NOT its visible centroid. So the cursor
        sprite extends down-right of (0, 0) by whatever its hotspot-
        to-centroid offset is — exactly the quantity we need to
        compensate for in every click.

        Per-attempt procedure:
          1. Cursor is at corner (caller just slammed).
          2. Capture pre-frame.
          3. Send a known nudge (+60 / +60 HID).
          4. Capture post-frame.
          5. abs-diff(pre, post). Two cursor blobs appear: the pre-
             position (near corner) and the post-position.
          6. The blob NEAREST origin is the cursor at the corner.
             Its centroid is the hotspot offset.
          7. Restore cursor to corner with -60 / -60.

        Runs the procedure THREE times and takes the median per
        axis. Single-shot measurements are noisy — observed
        variance across consecutive runs on the same setup was
        (0.19%, 0.75%) vs (1.10%, 2.36%) vs (1.09%, 2.35%), and
        the median is robust to one outlier without needing to
        understand WHY the outlier occurred (encoder hiccup,
        async camera vs HID timing, low-contrast cursor against
        a noisy background). Takes ~1.5s total for the calibration
        — paid once per session.
        """
        N_ATTEMPTS = 3
        NUDGE_HID = 60
        measurements: list[tuple[float, float, float]] = []  # (hx, hy, area)
        for attempt in range(N_ATTEMPTS):
            try:
                pre = await self._capture_gray()
                await self._send_hid(NUDGE_HID, NUDGE_HID)
                await asyncio.sleep(0.25)
                post = await self._capture_gray()
                await self._send_hid(-NUDGE_HID, -NUDGE_HID)
                await asyncio.sleep(0.15)

                diff = cv2.absdiff(pre, post)
                _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                )

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                h, w = pre.shape[:2]
                img_area = h * w
                best = None
                for c in contours:
                    a = cv2.contourArea(c)
                    if a < img_area * 0.00005 or a > img_area * 0.02:
                        continue
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        continue
                    cx = M["m10"] / M["m00"] / w
                    cy = M["m01"] / M["m00"] / h
                    d = math.hypot(cx, cy)
                    if best is None or d < best[0]:
                        best = (d, cx, cy, a / img_area)

                if best is None:
                    continue
                _, hx, hy, area = best
                # The nearest-origin blob must actually be near the
                # origin. The cursor at the corner sits ~half-cursor
                # inside the visible region, so realistic bounds are
                # well under 10%.
                if hx > 0.10 or hy > 0.10:
                    continue
                measurements.append((hx, hy, area))
            except Exception:
                continue

        if not measurements:
            print(
                "  Hotspot calibration: no usable measurements after "
                f"{N_ATTEMPTS} attempts — falling back to static "
                f"HOTSPOT_OFFSET_*_PCT constants."
            )
            return

        # Per-axis median is robust to a single outlier (e.g. one
        # attempt where the cursor barely moved and the "nearest
        # origin" was the corner itself rather than the cursor).
        xs = sorted(m[0] for m in measurements)
        ys = sorted(m[1] for m in measurements)
        areas = sorted(m[2] for m in measurements)
        n = len(measurements)
        med_x = xs[n // 2]
        med_y = ys[n // 2]
        med_area = areas[n // 2]

        self._calibrated_hotspot_offset = (med_x, med_y)
        self._cursor_area_pct = med_area
        spread_x = xs[-1] - xs[0] if n > 1 else 0.0
        spread_y = ys[-1] - ys[0] if n > 1 else 0.0
        print(
            f"  Hotspot offset calibrated at corner: "
            f"({med_x:.2%}, {med_y:.2%}) — median of {n} attempts "
            f"(spread x={spread_x:.2%}, y={spread_y:.2%}); "
            f"cursor area {med_area*100:.3f}% of frame."
        )

    def _hotspot_offset_pct(self) -> tuple[float, float]:
        """Adaptive hotspot offset based on the observed cursor size.

        For a cursor with area α (fraction of frame area) and aspect
        ratio matching the frame's W:H, the cursor's bounding box is
        ~sqrt(α · W/H) wide × sqrt(α · H/W) tall in frame-percent
        units. The arrow's hotspot is at the tip (top-left of the
        bounding box) while HSV detection centroids the visible
        blob, so the click lands ~half the bounding box down-right
        of the hotspot. Aiming the centroid (half_w, half_h) right
        and down of the operator's target compensates exactly.

        Returns the static HOTSPOT_OFFSET_*_PCT constants when no
        cursor area has been observed yet (first iteration before
        detection). For 1920×1080 frames with α=0.5%, this returns
        roughly (4.7%, 2.7%) — matching a max-size high-contrast
        cursor. For the default 32 px arrow (α≈0.05%) it returns
        about (1.5%, 0.8%), close to the legacy constants.
        """
        # Prefer the corner-clamp calibration when it's available —
        # it's measured directly from the running cursor shape and
        # is dead-accurate. The area-derived estimate is the next
        # best (works for any cursor shape but only approximates the
        # hotspot location). Static constants are the last resort.
        if self._calibrated_hotspot_offset is not None:
            return self._calibrated_hotspot_offset
        if self._cursor_area_pct is None:
            return HOTSPOT_OFFSET_X_PCT, HOTSPOT_OFFSET_Y_PCT
        # Use the frame's true aspect from session capture if
        # available; else default to 16:9.
        aspect = 1920.0 / 1080.0
        try:
            cap = self._session._capture
            w = getattr(cap, "width", None)
            h = getattr(cap, "height", None)
            if w and h:
                aspect = float(w) / float(h)
        except Exception:
            pass
        area = max(0.0, self._cursor_area_pct)
        # Half the bounding-box width / height in image-percent.
        # Floor with the legacy constants so a tiny observed blob
        # (probably a partial detection) doesn't reduce the offset
        # below what the default cursor needs.
        half_w = 0.5 * math.sqrt(area * aspect)
        half_h = 0.5 * math.sqrt(area / aspect)
        return (
            max(HOTSPOT_OFFSET_X_PCT, half_w),
            max(HOTSPOT_OFFSET_Y_PCT, half_h),
        )

    async def _send_hid(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        await self._session._send_hid_moves(dx, dy)

    async def _send_hid_burst(self, dx: int, dy: int) -> None:
        """Cruise-mode HID send. One round-trip to the Pi (via
        ``mouse.move_large``) which splits into ±127 chunks
        server-side and ships them back-to-back with no inter-
        chunk delay. Order-of-magnitude faster than the chunked
        closed-loop path on remote (USB-ECM / BT) transports
        where per-chunk HTTP overhead dominates. Falls back to
        the chunked path on backends that don't ship the burst
        endpoint.
        """
        if dx == 0 and dy == 0:
            return
        mouse = getattr(self._session._ctx, "mouse", None)
        if mouse is None:
            return
        try:
            await mouse.move_large(dx, dy)
        except Exception:
            # Fall back to chunked send if Pi doesn't recognise
            # the burst endpoint (older firmware).
            await self._session._send_hid_moves(dx, dy)

    def _refine_fast_ratio(
        self,
        hid_dx: int, hid_dy: int,
        measured_dx_pct: float, measured_dy_pct: float,
    ) -> None:
        """EMA-update ``_pct_per_hid_fast_*`` from observed cruise
        deltas. Same direction/magnitude guards as ``_refine_ratio``
        but writes the cruise-specific ratios so the chunked closed-
        loop calibration stays untouched.
        """
        FAST_RATIO_MAX = RATIO_MAX * 10
        for hid, meas, axis in (
            (hid_dx, measured_dx_pct, "x"),
            (hid_dy, measured_dy_pct, "y"),
        ):
            if abs(hid) < MIN_HID_PER_AXIS or meas == 0:
                continue
            if (hid > 0) != (meas > 0):
                continue
            current = (self._pct_per_hid_fast_x if axis == "x"
                       else self._pct_per_hid_fast_y)
            expected = abs(hid) * current
            if abs(meas) < 0.05 * expected:
                continue
            obs = abs(meas) / abs(hid)
            if not (RATIO_MIN <= obs <= FAST_RATIO_MAX):
                continue
            new = RATIO_EMA * obs + (1 - RATIO_EMA) * current
            new = max(RATIO_MIN, min(FAST_RATIO_MAX, new))
            if axis == "x":
                self._pct_per_hid_fast_x = new
            else:
                self._pct_per_hid_fast_y = new

    async def _capture_gray(self) -> np.ndarray:
        frame = await self._session._capture.capture_frame()
        self._record_session_frame(frame.image, "homer_capture")
        return cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

    async def _capture_color(self) -> np.ndarray:
        frame = await self._session._capture.capture_frame()
        self._record_session_frame(frame.image, "homer_capture")
        return frame.image

    def _record_session_frame(self, image, label: str) -> None:
        """Best-effort: persist into the session's flat output dir if
        the session adapter exposes one. Doesn't raise on failure."""
        session_out = getattr(self._session, "output_dir", None)
        if session_out is None or image is None:
            return
        ctx = getattr(self._session, "_ctx", None)
        # Prefer the AgentContext.record_frame helper for consistent
        # sequential numbering; fall back to a direct write.
        try:
            if ctx is not None and hasattr(ctx, "record_frame"):
                ctx.record_frame(image, label=label)
            else:
                import time as _t
                fname = f"{int(_t.time()*1000)}_{label}.png"
                cv2.imwrite(str(session_out / fname), image)
        except Exception:
            pass

    async def _verify_hsv_by_motion(
        self,
        candidate: tuple[float, float],
        run_dir: Path,
    ) -> tuple[float, float] | None:
        """Confirm a HSV candidate is the cursor by nudging and re-detecting.

        Send a known horizontal nudge; if HSV finds the candidate at
        roughly the new expected position, it's the cursor. If it
        stayed put, it was a static red UI element.
        """
        nudge_hid = 80
        await self._send_hid(nudge_hid, 0)
        await asyncio.sleep(SETTLE_SEC + 0.10)
        post = await self._capture_color()
        new_hit = find_cursor_hsv(post)
        if new_hit is None:
            return None
        # Expected: candidate moved right by ~nudge_hid * ratio in image space
        expected_dx = nudge_hid * self._pct_per_hid_x
        observed_dx = new_hit.x_pct - candidate[0]
        observed_dy = new_hit.y_pct - candidate[1]
        # Accept if observed moved ≥ 30% of expected in the same direction,
        # AND barely moved vertically.
        if observed_dx < expected_dx * 0.3:
            return None
        if abs(observed_dy) > 0.05:
            return None
        return (new_hit.x_pct, new_hit.y_pct)

    async def _find_cursor_via_oscillation(
        self, run_dir: Path, label: str = "init",
    ) -> tuple[float, float] | None:
        """Locate cursor by jiggling and finding the high-variance cluster.

        Sends an oscillation pattern, captures one frame per step, and
        lets ``find_cursor_by_variance`` pick the moving cluster.
        Robust to cursor color/shape — only requires the cursor exists
        and the rest of the screen is roughly static.

        Two-attempt cascade. The default amplitudes (±20 / ±40 HID)
        work fine for a redglass cursor on a webcam-pointed-at-Ubuntu
        target, but they're too small on a screen-share virtual camera
        looking at a remote macOS desktop: the default macOS arrow is
        ~15 px and low-contrast, and a 20-40 HID jiggle leaves a
        variance footprint smaller than the per-frame encoder noise.
        If the small shake fails, retry with ~3× amplitude and a
        looser variance threshold + min-active-pixels gate — same
        idea a human uses when they can't find their cursor and start
        waving the mouse harder.
        """
        # (amplitudes, variance_threshold, min_active_pixels) per attempt
        attempts = [
            # Original profile — fast, low false-positive rate.
            ([(20, 0), (-40, 0), (40, 0), (0, 20), (0, -40), (0, 40)],
             8.0, 30),
            # Big-shake fallback — covers 3× more screen, accepts a
            # fainter variance signature (encoder noise on a lossy
            # screen-share stream sits around 3-5 std-dev per pixel;
            # the cursor's true motion sits above that).
            ([(60, 0), (-120, 0), (120, 0), (0, 60), (0, -120), (0, 120)],
             4.0, 12),
        ]
        result = None
        for idx, (oscillation, var_thr, min_px) in enumerate(attempts):
            frames: list[np.ndarray] = []
            frames.append(await self._capture_gray())
            for dx, dy in oscillation:
                await self._send_hid(dx, dy)
                await asyncio.sleep(0.10)
                frames.append(await self._capture_gray())
            result = find_cursor_by_variance(
                frames,
                variance_threshold=var_thr,
                min_active_pixels=min_px,
                return_area=True,
            )
            if result is not None:
                # Record the oscillation-derived area so the
                # adaptive hotspot offset works even when HSV's
                # red threshold rejects the on-screen cursor (the
                # common case on a screen-share remote target).
                self._cursor_area_pct = result[2]
                print(
                    f"  Oscillation area: {result[2]*100:.3f}% of frame"
                )
                # Strip the area for downstream callers that expect
                # the historical (x, y) tuple shape.
                result = (result[0], result[1])
                if idx > 0:
                    print(
                        f"  Cursor found by oscillation on attempt "
                        f"#{idx + 1} (bigger shake, var_thr={var_thr})"
                    )
                break
            if idx + 1 < len(attempts):
                print(
                    f"  Oscillation attempt #{idx + 1} found nothing — "
                    "shaking harder."
                )
        if run_dir is not None:
            try:
                # Save the variance map for debugging.
                arr = np.stack([f.astype(np.float32) for f in frames], axis=0)
                var = arr.std(axis=0)
                vmax = float(var.max()) if var.size else 1.0
                vis = (var / max(vmax, 1.0) * 255).astype(np.uint8)
                cv2.imwrite(
                    str(run_dir / f"oscillation_{label}_variance.png"),
                    vis,
                )
                if result is not None:
                    h, w = frames[0].shape[:2]
                    annotated = cv2.cvtColor(frames[-1], cv2.COLOR_GRAY2BGR)
                    cx, cy = int(result[0] * w), int(result[1] * h)
                    cv2.circle(annotated, (cx, cy), 24, (0, 255, 0), 2)
                    cv2.imwrite(
                        str(run_dir / f"oscillation_{label}_hit.png"),
                        annotated,
                    )
            except Exception:
                pass
        return result


    async def _verify_navigation(
        self,
        target_desc: str,
        pre_click_color: np.ndarray,
        run_dir: Path,
        step: int,
    ) -> tuple[bool, str]:
        """Decide whether the click actually navigated/activated the target.

        Captures a post-click frame after a longer wait (page-load
        time), then OCRs the URL bar and top page strip looking for
        the target's keywords. Uses tesseract (reliable for URL/title
        text) before falling back to gemma.
        """
        # Allow time for SPA navigation + page render.
        await asyncio.sleep(2.5)
        try:
            post = await self._capture_color()
        except Exception as e:
            return False, f"post_capture_failed: {e}"
        try:
            cv2.imwrite(
                str(run_dir / f"step_{step:02d}_postclick_full.png"), post,
            )
            cv2.imwrite(
                str(run_dir / f"step_{step:02d}_preclick_full.png"),
                pre_click_color,
            )
        except Exception:
            pass

        h, w = post.shape[:2]
        # Oracle keyword priority: quoted target text wins. Otherwise
        # generic words like "subreddit" would let any subreddit
        # confirm any click.
        import re as _re
        quoted_oracle = _re.findall(r"['\"]([^'\"]+)['\"]", target_desc)
        if quoted_oracle:
            keywords = [q.lower() for q in quoted_oracle]
        else:
            keywords = ClosedLoopHomer._target_keywords(target_desc)

        # OCR the URL bar strip first — the URL is the most reliable
        # navigation indicator. Crop top 8% with both polarities.
        urlbar = post[int(h * 0.0):int(h * 0.10), :]
        page_strip = post[int(h * 0.05):int(h * 0.40), :]
        try:
            cv2.imwrite(str(run_dir / f"step_{step:02d}_urlbar.png"), urlbar)
            cv2.imwrite(
                str(run_dir / f"step_{step:02d}_titlestrip.png"), page_strip,
            )
        except Exception:
            pass

        if have_ocr():
            for region_name, region in (("urlbar", urlbar), ("title", page_strip)):
                hits = ocr_find_text(region, keywords)
                if hits:
                    top = hits[0]
                    print(
                        f"  Oracle (OCR/{region_name}): matched "
                        f"{top.text!r} conf={top.confidence:.0f}"
                    )
                    return True, (
                        f"OCR found {top.text!r} in {region_name} "
                        f"(conf={top.confidence:.0f})"
                    )

            # Diagnostic: dump full OCR text from URL bar and strip.
            try:
                import pytesseract  # type: ignore
                from handsneyes.core.vision.ocr_finder import (
                    _preprocess_for_ocr,
                )
                url_norm = pytesseract.image_to_string(urlbar)
                url_inv = pytesseract.image_to_string(
                    _preprocess_for_ocr(urlbar, scale=4, invert=True),
                )
                title_norm = pytesseract.image_to_string(page_strip)
                title_inv = pytesseract.image_to_string(
                    _preprocess_for_ocr(page_strip, scale=4, invert=True),
                )
                (run_dir / f"step_{step:02d}_oracle_ocr.txt").write_text(
                    f"--- URL BAR (normal) ---\n{url_norm}\n"
                    f"--- URL BAR (inverted) ---\n{url_inv}\n"
                    f"--- TITLE STRIP (normal) ---\n{title_norm}\n"
                    f"--- TITLE STRIP (inverted) ---\n{title_inv}\n"
                )
                print(
                    f"  Oracle: OCR found nothing matching {keywords}. "
                    f"Full OCR dump saved."
                )
            except Exception as e:
                print(f"  Oracle OCR diagnostic failed: {e}")

        # Gemma fallback — describe the page.
        b64 = await self._encode(page_strip)
        await self._session._ensure_client()
        prompt = (
            "You are a JSON API. The image is the top portion of a web "
            "page (browser chrome + page header). Read out any visible "
            "URL, page title, heading, breadcrumb, or community name.\n\n"
            "Respond with ONLY a JSON object — no preamble, no markdown.\n\n"
            'Schema: {"url": "<text in browser address bar, '
            'verbatim>", "title_text": "<most prominent heading>", '
            '"all_text": "<every word visible, space-separated>"}'
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": "Read the page. Reply JSON only."},
                ],
            },
        ]
        try:
            resp = await self._session._client.chat.completions.create(
                model=self._session._model,
                max_tokens=400,
                temperature=0.0,
                messages=messages,
            )
            raw = self._session._evaluator._best_text_from_response(resp) or ""
            data = self._session._evaluator._extract_json(raw) or {}
            url = str(data.get("url", "")).strip()
            title = str(data.get("title_text", "")).strip()
            all_text = str(data.get("all_text", "")).strip()
            combined = f"{url} {title} {all_text}".lower()
        except Exception as e:
            return False, f"oracle_query_failed: {e}"

        import re as _re
        matched = [
            k for k in keywords
            if _re.search(rf"\b{_re.escape(k)}\b", combined)
        ]
        for k in keywords:
            if _re.search(rf"r/{_re.escape(k)}", combined):
                if k not in matched:
                    matched.append(f"r/{k}")

        print(
            f"  Oracle (gemma): url={url!r} title={title!r} "
            f"keywords={keywords} matched={matched}"
        )
        if matched:
            return True, (
                f"page mentions {matched} (url='{url}', title='{title}')"
            )
        return False, (
            f"page does NOT contain target keywords {keywords} "
            f"(url='{url}', title='{title}')"
        )

    @dataclass
    class _Frame:
        gray: np.ndarray
        color: np.ndarray

    async def _capture_gray_and_color(self) -> "VisualServoHomer._Frame":
        frame = await self._session._capture.capture_frame()
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        return VisualServoHomer._Frame(gray=gray, color=frame.image)

    @staticmethod
    async def _encode(image_color: np.ndarray) -> str:
        resized = resize_for_mllm(
            enhance_for_screen(image_color),
            max_dimension=1280, min_dimension=768,
        )
        return numpy_to_base64_png(resized)

    async def _capture_proof(
        self, run_dir: Path, step: int,
    ) -> str | None:
        await asyncio.sleep(0.25)
        try:
            frame = await self._session._capture.capture_frame()
            path = run_dir / f"step_{step:02d}_after_click.png"
            cv2.imwrite(str(path), frame.image)
            return str(path)
        except Exception:
            return None

    def _dump_step_color(
        self,
        run_dir: Path,
        step: int,
        post_color: np.ndarray,
        cursor_img: tuple[float, float] | None,
        target_img: tuple[float, float] | None,
        rec: StepRecord,
    ) -> str | None:
        try:
            out = post_color.copy()
            h, w = out.shape[:2]
            if target_img is not None:
                tx = int(target_img[0] * w)
                ty = int(target_img[1] * h)
                cv2.rectangle(
                    out, (tx - 30, ty - 18), (tx + 30, ty + 18),
                    (0, 0, 255), 2,
                )
                cv2.putText(
                    out, "TARGET", (tx + 32, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
            if cursor_img is not None:
                cx = int(cursor_img[0] * w)
                cy = int(cursor_img[1] * h)
                cv2.circle(out, (cx, cy), 22, (0, 255, 255), 2)
                cv2.putText(
                    out, "CURSOR(HSV)", (cx + 24, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                )
            label = (
                f"step {step:02d} hid=({rec.hid_dx:+d},{rec.hid_dy:+d}) "
                f"resid={rec.residual_pct:.2%}"
                if rec.residual_pct is not None
                else f"step {step:02d}"
            )
            cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(
                out, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
            path = run_dir / f"step_{step:02d}.png"
            cv2.imwrite(str(path), out)
            return str(path)
        except Exception as e:
            logger.debug("dump_step_color failed: %s", e)
            return None

    def _dump_step(
        self,
        run_dir: Path,
        step: int,
        post_gray: np.ndarray,
        cursor_img: tuple[float, float] | None,
        target_img: tuple[float, float] | None,
        rec: StepRecord,
    ) -> str | None:
        try:
            out = cv2.cvtColor(post_gray, cv2.COLOR_GRAY2BGR)
            h, w = out.shape[:2]
            if target_img is not None:
                tx = int(target_img[0] * w)
                ty = int(target_img[1] * h)
                cv2.rectangle(
                    out, (tx - 30, ty - 18), (tx + 30, ty + 18),
                    (0, 0, 255), 2,
                )
                cv2.putText(
                    out, "TARGET", (tx + 32, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
            if cursor_img is not None:
                cx = int(cursor_img[0] * w)
                cy = int(cursor_img[1] * h)
                cv2.circle(out, (cx, cy), 18, (0, 255, 255), 2)
                cv2.putText(
                    out, "CURSOR(seen)", (cx + 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                )
            label = (
                f"step {step:02d} hid=({rec.hid_dx:+d},{rec.hid_dy:+d}) "
                f"resid={rec.residual_pct:.2%}"
                if rec.residual_pct is not None
                else f"step {step:02d}"
            )
            cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(
                out, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
            path = run_dir / f"step_{step:02d}.png"
            cv2.imwrite(str(path), out)
            return str(path)
        except Exception as e:
            logger.debug("dump_step failed: %s", e)
            return None
