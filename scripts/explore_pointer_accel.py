#!/usr/bin/env python3
"""explore_pointer_accel.py — autonomous data collection for the
pointer_accel model. No clicking, no human in the loop, no
``click_at`` target lookup.

Loop:
  1. Locate cursor via oscillation-variance (6-frame jiggle).
  2. Sample a random (hid_dx, hid_dy) — direction uniform on the
     unit circle, magnitude log-uniform in [--mag-min, --mag-max].
     Sign-flip an axis if the predicted post-position would land
     off-screen (we use a coarse ratio estimate of 0.002 pct/hid).
  3. Send the HID delta. Settle.
  4. Re-locate cursor via oscillation-variance.
  5. Write one history.jsonl row in the same schema the homer
     produces, so build_pointer_accel_dataset.py picks it up.

Rows land in
``<runs_root>/explore_<ts>/homer/<id>/history.jsonl`` — that path
matches the builder's glob (``**/homer/*/history.jsonl``).

Each sample is ~3s wall time (two oscillations + a test move +
settle). ~1200 samples/hour. Run it for 30 min unattended → enough
data to retrain.

Usage::

    python scripts/explore_pointer_accel.py --samples 200
    python scripts/explore_pointer_accel.py --target couch-ubuntu \\
        --samples 1000 --mag-min 5 --mag-max 120

The script connects to the target's Pi via the existing HTTP
backend (which now hard-fails at startup if BT isn't paired —
see io/mouse/backends/http.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

# Make `src/` importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from handsneyes.core.capture.webcam import WebcamCapture  # noqa: E402
from handsneyes.core.vision.cursor_finder import (  # noqa: E402
    find_cursor_by_variance,
)
from handsneyes.io.mouse.backends.http import HttpMouseOutput  # noqa: E402
from handsneyes.targets import TargetRegistry  # noqa: E402

logger = logging.getLogger("explore_pointer_accel")

# Coarse pct-per-hid estimate. Used for predictive edge guarding
# only — the real ratio comes from the measurements we're collecting.
_ROUGH_RATIO = 0.002

# Mean-brightness threshold (0..1) below which we treat the target as
# asleep / locked. Matches WakeAgent's threshold.
_ASLEEP_BRIGHTNESS = 0.06

# Variance-jiggle pattern that mirrors VisualServoHomer's. Symmetric
# so the centroid of the variance trail ≈ original cursor position.
_OSCILLATION = [(20, 0), (-40, 0), (40, 0), (0, 20), (0, -40), (0, 40)]


async def _capture_gray(cap: WebcamCapture) -> np.ndarray:
    frame = await cap.capture_frame()
    img = frame.image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


async def _send(mouse: HttpMouseOutput, dx: int, dy: int) -> None:
    # Pi clamps each report to [-127, 127]; split if necessary.
    while abs(dx) > 120 or abs(dy) > 120:
        step_x = max(-120, min(120, dx))
        step_y = max(-120, min(120, dy))
        await mouse.move(step_x, step_y)
        await asyncio.sleep(0.02)
        dx -= step_x
        dy -= step_y
    if dx or dy:
        await mouse.move(dx, dy)


async def _locate_cursor(
    cap: WebcamCapture, mouse: HttpMouseOutput,
    *, debug_dir: Path | None = None, label: str = "",
) -> tuple[float, float] | None:
    """Oscillation-variance cursor detection. ~700ms wall time."""
    frames: list[np.ndarray] = [await _capture_gray(cap)]
    for dx, dy in _OSCILLATION:
        await _send(mouse, dx, dy)
        await asyncio.sleep(0.10)
        frames.append(await _capture_gray(cap))
    hit = find_cursor_by_variance(frames)
    if debug_dir is not None:
        # Persist the first frame + variance map so failures are
        # inspectable instead of opaque.
        try:
            arr = np.stack([f.astype(np.float32) for f in frames], axis=0)
            var = arr.std(axis=0)
            vmax = float(var.max()) if var.size else 1.0
            vis = (var / max(vmax, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"oscillation_{label}_frame0.png"), frames[0])
            cv2.imwrite(str(debug_dir / f"oscillation_{label}_variance.png"), vis)
        except Exception:
            pass
    return hit


async def _wake_target(
    cap: WebcamCapture, mouse: HttpMouseOutput,
) -> tuple[bool, float]:
    """Check brightness; if below threshold, jiggle the mouse. Returns
    (awake, brightness_after) where awake is True iff the post-wake
    brightness is above the asleep threshold.
    """
    frame = await cap.capture_frame()
    img = frame.image
    if img is None:
        return False, 0.0
    brightness = float(np.asarray(img).mean()) / 255.0
    if brightness >= _ASLEEP_BRIGHTNESS:
        return True, brightness
    logger.info(
        "screen looks asleep (brightness=%.3f) — jiggling", brightness,
    )
    for _ in range(6):
        try:
            await mouse.move(20, 0)
            await asyncio.sleep(0.05)
            await mouse.move(-20, 0)
            await asyncio.sleep(0.05)
        except Exception as e:  # noqa: BLE001
            logger.warning("wake jiggle failed: %s", e)
            return False, brightness
    await asyncio.sleep(0.8)
    frame = await cap.capture_frame()
    brightness = float(np.asarray(frame.image).mean()) / 255.0
    return brightness >= _ASLEEP_BRIGHTNESS, brightness


def _sample_hid(
    cursor: tuple[float, float],
    *,
    mag_min: int,
    mag_max: int,
    margin: float,
    rng: random.Random,
) -> tuple[int, int]:
    """Log-uniform magnitude × uniform direction. Sign-flips if the
    predicted destination would leave a [margin, 1-margin] safe band."""
    log_mag = rng.uniform(math.log(mag_min), math.log(mag_max))
    mag = math.exp(log_mag)
    theta = rng.uniform(0.0, 2 * math.pi)
    dx = int(round(mag * math.cos(theta)))
    dy = int(round(mag * math.sin(theta)))
    cx, cy = cursor
    pred_x = cx + dx * _ROUGH_RATIO
    pred_y = cy + dy * _ROUGH_RATIO
    if pred_x < margin or pred_x > 1 - margin:
        dx = -dx
    if pred_y < margin or pred_y > 1 - margin:
        dy = -dy
    return dx, dy


async def _run_session(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed if args.seed >= 0 else None)
    registry = TargetRegistry.load_default()
    if args.target:
        target = registry.get(args.target)
    else:
        non_headless = [
            t for t in registry.targets.values() if t.platform != "headless"
        ]
        if not non_headless:
            logger.error(
                "no non-headless target in registry — pass --target or "
                "configure config/targets.toml",
            )
            return 4
        target = non_headless[0]
    logger.info(
        "target=%s pi=%s cam=%d", target.name, target.pi_url, target.camera_index,
    )

    runs_root = (
        Path(args.runs_root) if args.runs_root
        else Path.home() / ".local/share/handsneyes/runs"
    )
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = runs_root / f"explore_{ts}"
    homer_id = f"explore-{uuid.uuid4().hex[:8]}"
    out_dir = session_dir / "homer" / homer_id
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    logger.info("writing rows to %s", history_path)

    cap = WebcamCapture(device_index=target.camera_index)
    await cap.open()
    mouse = HttpMouseOutput(
        base_url=target.pi_url, transport=target.transport,
    )
    await mouse.connect()

    kept = 0
    dropped = 0
    consecutive_failures = 0
    step_idx = 0
    t0 = time.time()
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    try:
        awake, brightness = await _wake_target(cap, mouse)
        if not awake:
            logger.error(
                "target screen still dark after wake jiggle "
                "(brightness=%.3f). Likely asleep or the webcam isn't "
                "pointed at it. First frame saved to %s.",
                brightness, debug_dir / "preflight_frame.png",
            )
            try:
                last_frame = await cap.capture_frame()
                cv2.imwrite(
                    str(debug_dir / "preflight_frame.png"), last_frame.image,
                )
            except Exception:
                pass
            return 2

        cursor = await _locate_cursor(
            cap, mouse, debug_dir=debug_dir, label="preflight",
        )
        if cursor is None:
            logger.error(
                "initial cursor location failed (brightness=%.3f). "
                "Frame + variance map written to %s — inspect to see "
                "whether the screen is locked, the cursor is hidden, "
                "or the webcam is misframed.",
                brightness, debug_dir,
            )
            return 2

        while kept < args.samples:
            dx, dy = _sample_hid(
                cursor,
                mag_min=args.mag_min, mag_max=args.mag_max,
                margin=args.edge_margin, rng=rng,
            )
            try:
                await _send(mouse, dx, dy)
            except Exception as e:  # noqa: BLE001
                logger.warning("send failed: %s — sleeping 2s", e)
                await asyncio.sleep(2.0)
                consecutive_failures += 1
                if consecutive_failures > 5:
                    logger.error("5 consecutive HID failures — aborting")
                    return 3
                continue
            await asyncio.sleep(args.settle)

            cursor_after = await _locate_cursor(cap, mouse)
            if cursor_after is None:
                dropped += 1
                consecutive_failures += 1
                if consecutive_failures >= 4:
                    logger.warning(
                        "%d misses in a row — pausing 2s to let target settle",
                        consecutive_failures,
                    )
                    await asyncio.sleep(2.0)
                    consecutive_failures = 0
                continue

            mdx = cursor_after[0] - cursor[0]
            mdy = cursor_after[1] - cursor[1]

            # Sanity gate: a detected delta should be in roughly the
            # same direction as the sent HID and within an order of
            # magnitude. Reject obviously corrupt rows so they don't
            # poison training.
            ok = True
            for h, m in ((dx, mdx), (dy, mdy)):
                if abs(h) < 3:
                    continue
                ratio = abs(m / h)
                if ratio < 1e-4 or ratio > 1.5e-2:
                    ok = False
                    break
                if (m * h) < 0 and abs(m) > 0.01:
                    # Opposite-direction motion >1% of screen — most
                    # likely a misdetection (variance picked up an
                    # unrelated moving region).
                    ok = False
                    break

            row = {
                "cursor_img": [cursor[0], cursor[1]],
                "target_img": [
                    cursor[0] + dx * _ROUGH_RATIO,
                    cursor[1] + dy * _ROUGH_RATIO,
                ],
                "hid_dx": dx, "hid_dy": dy,
                "measured_dx_pct": mdx, "measured_dy_pct": mdy,
                "ratio_x": (abs(mdx / dx) if abs(dx) >= 3 else None),
                "ratio_y": (abs(mdy / dy) if abs(dy) >= 3 else None),
                "note": "explore_oscillation",
                "ts": time.time(),
                "step_idx": step_idx,
                "sanity_ok": ok,
            }
            step_idx += 1
            if ok:
                with history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                kept += 1
                consecutive_failures = 0
            else:
                dropped += 1

            cursor = cursor_after
            if kept % 25 == 0 and kept:
                elapsed = time.time() - t0
                rate = kept / elapsed * 60
                logger.info(
                    "%d kept / %d dropped (%.1f rows/min)",
                    kept, dropped, rate,
                )

            # Every 50 kept samples, scatter the cursor with a few
            # large random bursts so we don't oversample one screen
            # region.
            if kept and kept % 50 == 0:
                for _ in range(3):
                    rdx = rng.randint(-80, 80)
                    rdy = rng.randint(-80, 80)
                    await _send(mouse, rdx, rdy)
                    await asyncio.sleep(0.1)
                cursor = await _locate_cursor(cap, mouse) or cursor
    except KeyboardInterrupt:
        logger.info("interrupted by user — flushed %d rows", kept)
    finally:
        try:
            await mouse.disconnect()
        except Exception:
            pass
        try:
            await cap.close()
        except Exception:
            pass

    elapsed = time.time() - t0
    print(
        f"\n=== explore summary ===\n"
        f"  kept:     {kept}\n"
        f"  dropped:  {dropped}\n"
        f"  elapsed:  {elapsed:.1f}s\n"
        f"  rate:     {(kept / elapsed * 60 if elapsed else 0):.1f} rows/min\n"
        f"  out:      {history_path}\n",
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--target", default=None,
        help="Target name from targets.toml. Defaults to the first "
             "non-headless target.",
    )
    ap.add_argument(
        "--samples", type=int, default=300,
        help="Number of accepted samples to collect.",
    )
    ap.add_argument(
        "--mag-min", type=int, default=5,
        help="Minimum HID magnitude per axis (inclusive).",
    )
    ap.add_argument(
        "--mag-max", type=int, default=120,
        help="Maximum HID magnitude per axis (inclusive).",
    )
    ap.add_argument(
        "--settle", type=float, default=0.20,
        help="Seconds to wait after the test HID before re-detecting.",
    )
    ap.add_argument(
        "--edge-margin", type=float, default=0.08,
        help="Refuse to predict a move that lands within this margin "
             "(fraction of screen) of any edge — sign-flip instead.",
    )
    ap.add_argument(
        "--runs-root", type=str, default=None,
        help="Override the runs-root output dir. Defaults to "
             "~/.local/share/handsneyes/runs/.",
    )
    ap.add_argument(
        "--seed", type=int, default=-1,
        help="RNG seed. -1 for non-deterministic.",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(_run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
