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

# Floor / ceiling on the learned HID-to-image ratio. Sane bounds for
# cursor acceleration on most desktops.
RATIO_MIN = 0.0006
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


def _persist_step(run_dir: Path, record: "StepRecord") -> None:
    """Append one step record as a JSONL row to
    ``<run_dir>/history.jsonl``. Best-effort; errors are swallowed
    so a logging failure can't break a live homer run.

    Each row is ``{ts, hid_dx, hid_dy, cursor_img: [x, y] | null,
    measured_dx_pct, measured_dy_pct, ratio_x, ratio_y, ...}`` — the
    fields a forward-model trainer needs to learn the OS-side
    pointer-acceleration curve from observed cursor deltas.
    """
    try:
        row = asdict(record)
        # ``asdict`` turns tuples into lists, which is fine for JSON.
        row["ts"] = time.time()
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
) -> None:
    """Append ``record`` to in-memory ``history`` *and* persist it.

    Used at every ``history.append(StepRecord(...))`` site so a
    single API change covers both the existing in-memory contract
    and the new on-disk JSONL training log.
    """
    history.append(record)
    _persist_step(run_dir, record)


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
    for cand in _LONGJUMP_CHECKPOINT_CANDIDATES:
        if (cand / "config.json").exists():
            try:
                m = LongJumpModel(cand)
                cal = _LONGJUMP_CALIBRATION.get(cand.name, (1.0, 1.0))
                m._calibration = cal  # consumed by predict_total_hid
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
    logger.warning(
        "VisualServoHomer: no long-jump checkpoint at %s. The homer "
        "will work but the slam-to-target first step uses ratio-only "
        "open-loop (each click takes ~3-5 more iterations than with "
        "long-jump). Train your own with scripts/collect_pointer_accel.sh "
        "+ scripts/train_longjump.py — see README "
        '"Calibrating the homer for your setup".',
        ", ".join(str(c) for c in _LONGJUMP_CHECKPOINT_CANDIDATES),
    )
    return None


def _try_load_pointer_accel():
    """Best-effort load of the trained open-loop pointer-accel MLP.

    Returns ``None`` if the package or checkpoints are missing — the
    homer falls back to a ratio-based open-loop seed (still works,
    just slower convergence).
    """
    try:
        from handsneyes.core.vision.pointer_accel import PointerAccelModel
    except Exception as e:
        logger.debug("pointer-accel module unavailable: %s", e)
        return None
    for cand in _POINTER_ACCEL_CHECKPOINT_CANDIDATES:
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
        "homer will work but convergence is slower. Train your own "
        "with scripts/collect_pointer_accel.sh + "
        "scripts/train_pointer_accel.py — see README "
        '"Calibrating the homer for your setup".',
        ", ".join(str(c) for c in _POINTER_ACCEL_CHECKPOINT_CANDIDATES),
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
        self._diff_misses_in_a_row = 0
        self._zoom_levels_applied = 0
        # If init's HSV candidate fails motion verification, disable
        # HSV for the rest of the run — there's a static red element
        # (Reddit logo, etc.) that would otherwise hijack every step's
        # detection and freeze the cursor estimate.
        self._hsv_enabled = False
        # Reuse the scene-map and target-keyword machinery from the
        # earlier homer — that part still works, the bug was elsewhere.
        self._helper = ClosedLoopHomer(session=session)
        # Optional open-loop pointer-acceleration model. Trained
        # offline (scripts/train_pointer_accel.py); used only as the
        # first-iteration seed of the servo loop. Closed-loop still
        # owns iterations 2+, so absence is harmless.
        self._pointer_accel = _try_load_pointer_accel()
        # Optional long-jump model for the slam-to-target first move.
        # Trained on aggregated trajectory data
        # (scripts/build_longjump_dataset.py + train_longjump.py).
        # When present, home_to_pixel fires a chain of back-to-back
        # HID bursts based on this model's prediction before entering
        # the closed-loop servo, cutting typical click time ~2.5x.
        self._longjump = _try_load_longjump()

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
        # corner where HSV may struggle near the edge.
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
        target_aim = (
            target_img[0] + HOTSPOT_OFFSET_X_PCT,
            target_img[1] + HOTSPOT_OFFSET_Y_PCT,
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
    ) -> ClickOutcome:
        """Run the visual-servo loop until cursor lands on target_aim.

        Extracted from ``_run_inner`` so the same loop can also serve
        manual click-to-pixel requests where the target is supplied
        directly rather than located via OCR/VLM. ``verify_navigation``
        gates the LLM URL-bar oracle: ``False`` skips both the navigation
        verify and the click-retry diamond pattern (click once and
        capture proof — fine for manual operator-driven clicks).

        ``axis_aligned``: per-iteration, send HID along ONLY the
        larger-residual axis (zero out the other). Used by the no-
        slam follow-up path so the cursor moves along axis-aligned
        L-shape paths instead of diagonals — diagonals sweep across
        sibling UI elements (the canonical case: clicking File then
        sweeping diagonally toward "Exit LibreOffice" hover-opens
        Edit, View, Insert on the way and the wrong menu ends up
        intercepting the click). Closes the residual along Y first
        for a tall move, then X (or vice versa) — the cursor stays
        in the column/row of the open UI element.
        """
        confirm_count = 0
        for step in range(1, MAX_STEPS + 1):
            t_step = time.monotonic()

            dx_pct = target_aim[0] - cursor_img[0]
            dy_pct = target_aim[1] - cursor_img[1]
            residual = math.hypot(dx_pct, dy_pct)

            if residual <= click_tol_pct:
                confirm_count += 1
                print(
                    f"  [{step:02d}] cursor=({cursor_img[0]:.2%},"
                    f"{cursor_img[1]:.2%}) aim=({target_aim[0]:.2%},"
                    f"{target_aim[1]:.2%}) residual={residual:.2%} ≤ "
                    f"{click_tol_pct:.1%} — confirm {confirm_count}/{confirm_frames}"
                )
                if confirm_count >= confirm_frames:
                    if not verify_navigation:
                        # Manual mode: geometric confirm is enough.
                        # Click once (unless caller opted out — used
                        # by /api/mouse/scroll, which homes the
                        # cursor to the hover position WITHOUT
                        # clicking, then scrolls at that pixel),
                        # capture proof, return success.
                        if click:
                            await self._session._executor._mouse.click(button)
                        try:
                            await asyncio.sleep(0.4)
                            proof = await self._capture_proof(
                                run_dir, step * 100,
                            )
                        except Exception:
                            proof = None
                        _record_step(run_dir, history, StepRecord(
                            cursor_img=cursor_img, target_img=target_img,
                            residual_pct=residual, hid_dx=0, hid_dy=0,
                            ratio_x=self._pct_per_hid_x,
                            ratio_y=self._pct_per_hid_y,
                            note="click_sent;manual_geometric_only",
                        ))
                        print(
                            f"  ✓ Geometric confirm — clicked {button} "
                            f"at ({target_aim[0]:.2%},{target_aim[1]:.2%})"
                        )
                        return ClickOutcome(
                            clicked=True, steps=step,
                            reason="geometric_confirm",
                            proof_path=proof, history=history,
                        )
                    # Click + verify. If verification fails, nudge
                    # through a small diamond pattern and retry — the
                    # cursor centroid is close but the click hotspot
                    # may be a few pixels off the link.
                    final_proof: str | None = None
                    final_reason: str = ""
                    final_ok = False
                    attempts: list[tuple[int, int]] = [
                        (0, 0)
                    ] + CLICK_RETRY_PATTERN_HID
                    for attempt_idx, (nx, ny) in enumerate(attempts):
                        if (nx, ny) != (0, 0):
                            await self._send_hid(nx, ny)
                            await asyncio.sleep(SETTLE_SEC)
                            print(
                                f"  Retry click {attempt_idx}/{len(attempts)-1} "
                                f"after nudge=({nx:+d},{ny:+d})"
                            )
                        pre_click_color = await self._capture_color()
                        await self._session._executor._mouse.click(button)
                        proof = await self._capture_proof(
                            run_dir, step * 100 + attempt_idx,
                        )
                        nav_ok, nav_reason = await self._verify_navigation(
                            target_desc, pre_click_color, run_dir,
                            step * 100 + attempt_idx,
                        )
                        final_proof = proof
                        final_reason = nav_reason
                        if nav_ok:
                            final_ok = True
                            # Capture a confirmed proof frame AFTER
                            # the page has fully navigated, so the
                            # screenshot matches the oracle's verdict.
                            try:
                                await asyncio.sleep(0.6)
                                confirmed = await self._capture_color()
                                cp = run_dir / f"step_{step:02d}_CONFIRMED.png"
                                cv2.imwrite(str(cp), confirmed)
                                final_proof = str(cp)
                            except Exception:
                                pass
                            break
                    _record_step(run_dir, history, StepRecord(
                        cursor_img=cursor_img, target_img=target_img,
                        residual_pct=residual, hid_dx=0, hid_dy=0,
                        ratio_x=self._pct_per_hid_x,
                        ratio_y=self._pct_per_hid_y,
                        note=f"click_sent;nav_ok={final_ok};{final_reason[:80]}",
                    ))
                    if final_ok:
                        print(
                            f"  ✓ Click landed on target — navigation "
                            f"confirmed: {final_reason}"
                        )
                        return ClickOutcome(
                            clicked=True, steps=step,
                            reason=f"navigation_confirmed: {final_reason}",
                            proof_path=final_proof, history=history,
                        )
                    print(
                        f"  ✗ All {len(attempts)} click attempts missed — "
                        f"{final_reason}. Reporting failure."
                    )
                    return ClickOutcome(
                        clicked=False, steps=step,
                        reason=f"click_attempts_no_nav: {final_reason}",
                        proof_path=final_proof, history=history,
                    )
                await asyncio.sleep(SETTLE_SEC)
                # Re-measure cursor only if HSV is reliable. When HSV
                # is disabled, the cursor doesn't move during sleep
                # so trust the previous estimate. Using HSV when it
                # latches onto a static red element would reset the
                # cursor incorrectly.
                if self._hsv_enabled:
                    frame_now = await self._capture_color()
                    redet = find_cursor_hsv(frame_now)
                    if redet is not None:
                        cursor_img = (redet.x_pct, redet.y_pct)
                continue
            confirm_count = 0

            # First iteration: try the learned open-loop pointer-
            # acceleration model as the seed move. The model maps
            # (hid_dx, hid_dy, cursor_x, cursor_y) → measured delta
            # under the target OS's pointer-accel curve; its inverse
            # gives the HID delta most likely to land near the target
            # in one shot. The existing closed-loop ratio still owns
            # iterations 2+ when the seed is imperfect (median seed
            # error is ~2 px on val, 90th-percentile ~140 px), so
            # this is a strict speedup with no precision loss.
            hid_dx = None
            hid_dy = None
            if step == 1 and self._pointer_accel is not None:
                try:
                    hid_dx, hid_dy = self._pointer_accel.inverse(
                        target_dx_pct=dx_pct,
                        target_dy_pct=dy_pct,
                        cursor_x_pct=cursor_img[0],
                        cursor_y_pct=cursor_img[1],
                        initial_ratio_x=self._pct_per_hid_x,
                        initial_ratio_y=self._pct_per_hid_y,
                    )
                    print(
                        f"  [{step:02d}] open-loop seed "
                        f"via PointerAccelModel → hid=({hid_dx},{hid_dy})"
                    )
                except Exception as e:
                    logger.debug("pointer-accel inverse failed: %s", e)
                    hid_dx = hid_dy = None
            if hid_dx is None or hid_dy is None:
                # Compute HID move using current ratio (clamped) and a
                # hard cap so we never blast off-screen.
                hid_dx, hid_dy = self._hid_for_residual(dx_pct, dy_pct)
            if hid_dx == 0 and hid_dy == 0:
                hid_dx = MIN_HID_PER_AXIS if dx_pct > 0 else (
                    -MIN_HID_PER_AXIS if dx_pct < 0 else 0
                )
                hid_dy = MIN_HID_PER_AXIS if dy_pct > 0 else (
                    -MIN_HID_PER_AXIS if dy_pct < 0 else 0
                )

            # Axis-aligned mode: zero out the smaller-residual axis
            # so the cursor moves only along the larger axis this
            # iteration. Two iterations of this produce an L-shaped
            # path (e.g. down-then-right) instead of a diagonal sweep
            # that would hover-open sibling UI elements on the way.
            if axis_aligned:
                if abs(dx_pct) > abs(dy_pct):
                    hid_dy = 0
                else:
                    hid_dx = 0

            pre_color = await self._capture_color()
            await self._send_hid(hid_dx, hid_dy)
            await asyncio.sleep(SETTLE_SEC)
            post_color = await self._capture_color()

            measured_dx_pct: float | None = None
            measured_dy_pct: float | None = None

            # If HSV passed motion verification at init, use it per
            # step (it's the most reliable). Otherwise skip — it would
            # otherwise lock onto static red UI elements.
            new_pos: tuple[float, float] | None = None
            new_hit: CursorHit | None = None
            if self._hsv_enabled:
                # Position-aware HSV with a motion-proportional
                # search radius. Without this, a static red icon
                # near the target (e.g. a tooltip dot under an
                # adjacent menu item) could fall inside a fixed-
                # size search ring and hijack the cursor estimate.
                # Scale the allowed search radius by how far we
                # actually expected the cursor to move: a tiny
                # convergence step has tiny uncertainty so the
                # radius should be tight; a big slam has more model
                # error so the radius widens. The hard floor (1.5%)
                # absorbs jitter; the ceiling (4%) keeps us from
                # going wider than ``find_cursor_hsv_near``'s
                # default and re-introducing the original bug.
                # Differential red-mask + direction + shape: the
                # cursor is the only red thing on the host that
                # responds to our HID. Static UI red (icons, syntax
                # highlights, text-field selection) cancels out in
                # the pre/post subtraction. Among the "newly red"
                # blobs that survive, we add two more priors:
                #   - direction match: observed displacement from
                #     the previous cursor position must align with
                #     the HID we sent (cos similarity ≥ 0.5),
                #     killing newly-red regions caused by a popup
                #     appearing (they don't lie on our HID vector).
                #   - arrow-like shape: the redglass cursor is
                #     taller than wide and fairly solid; static UI
                #     red is usually round / wide / sparse.
                expected_motion_pct = (
                    hid_dx * self._pct_per_hid_x,
                    hid_dy * self._pct_per_hid_y,
                )
                expected_motion_mag = math.hypot(*expected_motion_pct)
                radius = max(0.02, min(0.06, 0.4 * expected_motion_mag + 0.015))
                new_hit = find_cursor_hsv_motion_directed(
                    pre_color, post_color,
                    cursor_pre_pct=cursor_img,
                    expected_motion_pct=expected_motion_pct,
                    max_dist_pct=radius,
                )
                if new_hit is not None:
                    new_pos = (new_hit.x_pct, new_hit.y_pct)
            if new_pos is None:
                # ROI-prior diff: we know roughly where the cursor
                # moved, so look for the changed region near
                # cursor_img + expected_motion.
                expected_new = (
                    cursor_img[0] + hid_dx * self._pct_per_hid_x,
                    cursor_img[1] + hid_dy * self._pct_per_hid_y,
                )
                new_pos = self._diff_locate_with_prior(
                    pre_color, post_color, cursor_img, expected_new,
                )
                if new_pos is not None:
                    self._diff_misses_in_a_row = 0
                else:
                    self._diff_misses_in_a_row += 1

            if new_pos is not None:
                measured_dx_pct = new_pos[0] - cursor_img[0]
                measured_dy_pct = new_pos[1] - cursor_img[1]
                cursor_img = new_pos
                self._refine_ratio(
                    hid_dx, hid_dy, measured_dx_pct, measured_dy_pct,
                )
            else:
                # Fall back to open-loop estimate; if we miss too many
                # in a row, force a re-localize via oscillation.
                cursor_img = (
                    cursor_img[0] + hid_dx * self._pct_per_hid_x,
                    cursor_img[1] + hid_dy * self._pct_per_hid_y,
                )
                if self._diff_misses_in_a_row >= 3:
                    print(
                        f"  [{step:02d}] 3 detection misses — "
                        f"re-localizing via oscillation."
                    )
                    relocated = await self._find_cursor_via_oscillation(
                        run_dir, label=f"step{step:02d}_relocate",
                    )
                    if relocated is not None:
                        cursor_img = relocated
                    self._diff_misses_in_a_row = 0

            elapsed = time.monotonic() - t_step
            measured_str = (
                f"meas=({measured_dx_pct:+.2%},{measured_dy_pct:+.2%})"
                if measured_dx_pct is not None
                else "meas=HSV_MISS"
            )
            print(
                f"  [{step:02d}] hid=({hid_dx:+4d},{hid_dy:+4d}) "
                f"cursor→({cursor_img[0]:.2%},{cursor_img[1]:.2%}) "
                f"aim=({target_aim[0]:.2%},{target_aim[1]:.2%}) "
                f"resid={residual:.2%} {measured_str} "
                f"ratio=({self._pct_per_hid_x*1000:.4f},{self._pct_per_hid_y*1000:.4f})‰ "
                f"{elapsed:.2f}s"
            )

            _record_step(run_dir, history, StepRecord(
                cursor_img=cursor_img, target_img=target_img,
                residual_pct=residual, hid_dx=hid_dx, hid_dy=hid_dy,
                measured_dx_pct=measured_dx_pct,
                measured_dy_pct=measured_dy_pct,
                ratio_x=self._pct_per_hid_x,
                ratio_y=self._pct_per_hid_y,
                note=("hsv_measured" if new_hit is not None
                      else "openloop_fallback"),
            ))
            last_proof = self._dump_step_color(
                run_dir, step, post_color, cursor_img, target_img,
                history[-1],
            )

        print(
            f"  Reached MAX_STEPS={MAX_STEPS} without geometric confirm. "
            "NOT clicking."
        )
        return ClickOutcome(
            clicked=False, steps=MAX_STEPS, reason="max_steps",
            proof_path=last_proof, history=history,
        )

    # ────────────────────── manual click-to-pixel ──────────────────────

    async def home_to_pixel(
        self,
        x_pct: float,
        y_pct: float,
        button: str = "left",
        *,
        hotspot_offset: bool = True,
        click: bool = True,
        prev_cursor_pct: tuple[float, float] | None = None,
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

        target_img = (float(x_pct), float(y_pct))
        if hotspot_offset:
            target_aim = (
                target_img[0] + HOTSPOT_OFFSET_X_PCT,
                target_img[1] + HOTSPOT_OFFSET_Y_PCT,
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
        #
        # confirm_frames=3 requires three consecutive in-tolerance
        # readings before clicking. With per-reading detector noise
        # σ≈5 px, a single in-tolerance reading at ~4 px gate fires
        # ~25% of the time by luck; three in a row drops the
        # false-confirm rate to ~1%. Costs ~6s in extra detections
        # per click, buys you precision that lands on small buttons
        # instead of next to them.
        #
        # In no-slam mode (chained clicks within an open UI) we
        # also constrain the closed loop to axis-aligned moves so
        # diagonal transit doesn't open sibling menus.
        return await self._servo_loop(
            target_aim=target_aim, target_img=target_img,
            cursor_img=cursor_img, button=button, run_dir=run_dir,
            history=history, target_desc="<manual>",
            verify_navigation=False, last_proof=None,
            confirm_frames=3,
            click_tol_pct=0.004,
            click=click,
            axis_aligned=(prev_cursor_pct is not None),
        )

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
        ))
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

    async def _redetect_cursor(
        self, prev_gray: np.ndarray, last_known: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Sanity-check cursor location via a small jiggle-and-diff.

        Used during the click confirm gate to make sure the cursor
        hasn't drifted between the two confirm frames.
        """
        await self._send_hid(20, 0)
        await asyncio.sleep(SETTLE_SEC)
        mid = await self._capture_gray()
        await self._send_hid(-20, 0)
        await asyncio.sleep(SETTLE_SEC)
        end = await self._capture_gray()
        new_cursor, _ = self._detect_cursor_motion(
            prev_gray, mid,
            expected_dx_pct=20 * self._pct_per_hid_x,
            expected_dy_pct=0,
            last_known=last_known,
        )
        del end  # we end at ~= prev cursor position
        return new_cursor

    # ────────────────────── frame diff plumbing ──────────────────────

    @staticmethod
    def _diff_blobs(
        a: np.ndarray, b: np.ndarray,
    ) -> list[dict]:
        """Return changed-region blobs as dicts of cx, cy, area (all
        in image-percent units 0..1)."""
        if a.shape != b.shape:
            return []
        diff = cv2.absdiff(a, b)
        _, thresh = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (DIFF_DILATE_KERNEL, DIFF_DILATE_KERNEL),
        )
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        h, w = a.shape[:2]
        img_area = h * w
        out: list[dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area * BLOB_MIN_AREA:
                continue
            if area > img_area * BLOB_MAX_AREA:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
            out.append({
                "cx": cx, "cy": cy,
                "area": area / img_area,
            })
        return out

    def _detect_cursor_motion(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        expected_dx_pct: float,
        expected_dy_pct: float,
        last_known: tuple[float, float],
        run_dir: Path | None = None,
        step: int | None = None,
    ) -> tuple[tuple[float, float] | None, list[dict]]:
        """Locate the cursor's NEW position by diffing pre→post.

        Strategy: find all changed blobs. The cursor diff has two
        signatures — an OLD (cursor was here) blob near ``last_known``
        and a NEW (cursor is here) blob near ``last_known +
        expected_motion``. Score each blob by how well it matches the
        expected NEW position. Return the best.
        """
        blobs = self._diff_blobs(pre, post)
        if run_dir is not None and step is not None:
            self._save_diff_debug(run_dir, f"step{step:02d}", pre, post, blobs)
        if not blobs:
            return None, blobs

        target_x = last_known[0] + expected_dx_pct
        target_y = last_known[1] + expected_dy_pct
        # Score: distance to expected NEW position; smaller = better.
        # Tie-break by direction agreement with motion vector from
        # last_known.
        scored: list[tuple[float, dict]] = []
        for b in blobs:
            d_new = math.hypot(b["cx"] - target_x, b["cy"] - target_y)
            scored.append((d_new, b))
        scored.sort(key=lambda t: t[0])
        best = scored[0][1]

        # Reject if best is implausibly far from expected new position
        # AND from last_known.
        d_to_new = math.hypot(best["cx"] - target_x, best["cy"] - target_y)
        d_to_last = math.hypot(best["cx"] - last_known[0], best["cy"] - last_known[1])
        if d_to_new > 0.20 and d_to_last > 0.30:
            return None, blobs

        return (best["cx"], best["cy"]), blobs

    @staticmethod
    def _save_diff_debug(
        run_dir: Path, label: str,
        pre: np.ndarray, post: np.ndarray,
        blobs: list[dict],
    ) -> None:
        try:
            diff = cv2.absdiff(pre, post)
            _, thresh = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
            vis = cv2.cvtColor(post, cv2.COLOR_GRAY2BGR)
            h, w = post.shape[:2]
            mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            mask[:, :, 0] = 0
            mask[:, :, 1] = 0
            vis = cv2.addWeighted(vis, 1.0, mask, 0.5, 0)
            for b in blobs:
                px = int(b["cx"] * w)
                py = int(b["cy"] * h)
                cv2.circle(vis, (px, py), 18, (0, 255, 255), 2)
            cv2.imwrite(str(run_dir / f"diff_{label}.png"), vis)
        except Exception:
            pass

    # ────────────────────── HID maths ──────────────────────

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
            if abs(meas) < 0.25 * expected:
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

    async def _slam_to_corner(self) -> None:
        print("  Slamming to top-left corner...")
        for _ in range(200):
            try:
                await self._session._executor._mouse.move(-20, -20)
            except Exception:
                pass
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.3)

    async def _send_hid(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        await self._session._send_hid_moves(dx, dy)

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

        Sends a small oscillation pattern, captures ~6 frames during,
        and lets ``find_cursor_by_variance`` pick the moving cluster.
        Robust to cursor color/shape — only requires the cursor exists
        and the rest of the screen is roughly static.
        """
        frames: list[np.ndarray] = []
        # Take an initial frame, then oscillate, capturing each step.
        frames.append(await self._capture_gray())
        # Small symmetric oscillation: trajectory returns to start so
        # the variance trail's centroid ≈ the original cursor pos.
        oscillation = [(20, 0), (-40, 0), (40, 0), (0, 20), (0, -40), (0, 40)]
        for dx, dy in oscillation:
            await self._send_hid(dx, dy)
            await asyncio.sleep(0.10)
            frames.append(await self._capture_gray())
        result = find_cursor_by_variance(frames)
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

    def _diff_locate_with_prior(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        prev_cursor: tuple[float, float],
        expected_new: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Find the cursor's new position by diffing pre→post within
        an ROI around the expected new position.

        Strategy: the diff has changed pixels near the OLD cursor (now
        empty) and near the NEW cursor (now occupied). Inside an ROI
        centred at ``expected_new``, find the largest changed blob; its
        centroid is the new cursor position.
        """
        if pre.ndim == 3:
            pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        if post.ndim == 3:
            post = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
        h, w = pre.shape[:2]

        # ROI radius: 20% of image (large enough to absorb ratio error
        # but tight enough to reject distant noise).
        roi_r = 0.20
        ex, ey = expected_new
        # Always include prev_cursor's vicinity too, so we don't miss
        # the case where the cursor barely moved.
        cx, cy = prev_cursor
        x0 = int(max(0, min(ex, cx) - roi_r) * w)
        y0 = int(max(0, min(ey, cy) - roi_r) * h)
        x1 = int(min(1, max(ex, cx) + roi_r) * w)
        y1 = int(min(1, max(ey, cy) + roi_r) * h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None

        roi_pre = pre[y0:y1, x0:x1]
        roi_post = post[y0:y1, x0:x1]
        diff = cv2.absdiff(roi_pre, roi_post)
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        roi_area = (x1 - x0) * (y1 - y0)
        # Score each blob by closeness to expected_new (in absolute
        # image coords) and require it not to be the OLD cursor blob.
        ex_px = ex * w
        ey_px = ey * h
        cx_px = cx * w
        cy_px = cy * h
        best = None
        best_d = float("inf")
        for c in contours:
            area = cv2.contourArea(c)
            if area < roi_area * 0.0005 or area > roi_area * 0.10:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            blob_x = M["m10"] / M["m00"] + x0
            blob_y = M["m01"] / M["m00"] + y0
            d_new = math.hypot(blob_x - ex_px, blob_y - ey_px)
            d_old = math.hypot(blob_x - cx_px, blob_y - cy_px)
            # Prefer blobs near expected new and far from old.
            score = d_new - 0.3 * d_old
            if score < best_d:
                best_d = score
                best = (blob_x / w, blob_y / h)
        return best

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
