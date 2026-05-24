#!/usr/bin/env python3
"""collect_cursor_detector_labels.py — (frame, cursor_xy) pairs for
training a single-frame cursor detector.

The current pipeline detects the Yaru white cursor with a 6-frame
oscillation pattern + variance centroid — ~3 captures per measurement
and ±3-5 px label noise. That floor caps every pointer_accel retrain.
A single-frame learned detector removes both: one capture per
measurement, sub-pixel accuracy (if the model is good).

This script generates the training set. Each iteration:

  1. Scatter the cursor with a random HID move (so consecutive
     samples don't cluster).
  2. Capture ONE frame F. This is the model's eventual input.
  3. Run the existing oscillation-variance detector to get the
     ground-truth cursor position P.
  4. Save F to disk + append {frame, x_pct, y_pct, ts} to labels.jsonl.

Output schema (so the trainer can consume it directly)::

    data/ml/cursor_detector/<ts>/frames/0000.png  (full-resolution webcam frame)
    data/ml/cursor_detector/<ts>/labels.jsonl     ({frame, x_pct, y_pct, ts, brightness})

Throughput: ~5s per labeled sample (oscillation dominates). ~720
samples/hour. Aim for 3000-5000 labels for a robust model.

Usage::

    python scripts/collect_cursor_detector_labels.py --samples 200
    python scripts/collect_cursor_detector_labels.py --samples 5000 --seed 7
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
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from handsneyes.core.capture.webcam import WebcamCapture  # noqa: E402
from handsneyes.core.vision.cursor_finder import (  # noqa: E402
    find_cursor_by_variance,
)
from handsneyes.io.mouse.backends.http import HttpMouseOutput  # noqa: E402
from handsneyes.targets import TargetRegistry  # noqa: E402

logger = logging.getLogger("collect_cursor")

_ASLEEP_BRIGHTNESS = 0.06
# Oscillation pattern matches VisualServoHomer's; symmetric so the
# variance centroid recovers the cursor's start position.
_OSCILLATION = [(20, 0), (-40, 0), (40, 0), (0, 20), (0, -40), (0, 40)]


async def _capture_gray(cap: WebcamCapture) -> np.ndarray:
    frame = await cap.capture_frame()
    img = frame.image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


async def _capture_color(cap: WebcamCapture) -> np.ndarray:
    frame = await cap.capture_frame()
    return frame.image


async def _send(mouse: HttpMouseOutput, dx: int, dy: int) -> None:
    while abs(dx) > 120 or abs(dy) > 120:
        sx = max(-120, min(120, dx))
        sy = max(-120, min(120, dy))
        await mouse.move(sx, sy)
        await asyncio.sleep(0.02)
        dx -= sx
        dy -= sy
    if dx or dy:
        await mouse.move(dx, dy)


async def _locate_cursor(
    cap: WebcamCapture, mouse: HttpMouseOutput,
) -> tuple[float, float] | None:
    frames: list[np.ndarray] = [await _capture_gray(cap)]
    for dx, dy in _OSCILLATION:
        await _send(mouse, dx, dy)
        await asyncio.sleep(0.10)
        frames.append(await _capture_gray(cap))
    return find_cursor_by_variance(frames)


def _looks_like_test_pattern(img: np.ndarray) -> bool:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return False
    vert = float(np.abs(np.diff(gray.astype(np.int16), axis=0)).mean())
    return vert < 1.5


async def _wake_target(
    cap: WebcamCapture, mouse: HttpMouseOutput,
) -> tuple[bool, float]:
    frame = await cap.capture_frame()
    img = frame.image
    if img is None:
        return False, 0.0
    if _looks_like_test_pattern(img):
        logger.error(
            "webcam is on the SMPTE test pattern — camera busy/missing. "
            "Replug or free it before retrying.",
        )
        return False, -1.0
    brightness = float(np.asarray(img).mean()) / 255.0
    if brightness >= _ASLEEP_BRIGHTNESS:
        return True, brightness
    logger.info("brightness=%.3f below threshold; jiggling", brightness)
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


def _safe_scatter(
    cursor: tuple[float, float], rng: random.Random,
    *, margin: float = 0.08,
) -> tuple[int, int]:
    """Pick a random HID move that keeps the cursor inside [margin, 1-margin].

    Magnitude is uniform in [20, 80] pixels — large enough to land in
    a fresh region of screen, small enough that the cursor's new
    background is varied across iterations."""
    mag = rng.uniform(20, 80)
    theta = rng.uniform(0.0, 2 * math.pi)
    dx = int(round(mag * math.cos(theta)))
    dy = int(round(mag * math.sin(theta)))
    cx, cy = cursor
    rough = 0.0008
    if cx + dx * rough < margin or cx + dx * rough > 1 - margin:
        dx = -dx
    if cy + dy * rough < margin or cy + dy * rough > 1 - margin:
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
            logger.error("no non-headless target configured")
            return 4
        target = non_headless[0]
    logger.info(
        "target=%s pi=%s cam=%d", target.name, target.pi_url, target.camera_index,
    )

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_root) / ts
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    logger.info("writing to %s", out_dir)

    cap = WebcamCapture(device_index=target.camera_index)
    await cap.open()
    mouse = HttpMouseOutput(
        base_url=target.pi_url, transport=target.transport,
    )
    await mouse.connect()

    kept = 0
    missed = 0
    t0 = time.time()
    try:
        awake, brightness = await _wake_target(cap, mouse)
        if not awake:
            logger.error("screen not usable (brightness=%.3f)", brightness)
            return 2

        cursor = await _locate_cursor(cap, mouse)
        if cursor is None:
            logger.error("initial cursor detect failed")
            return 2

        for idx in range(args.samples * 2):  # headroom for misses
            if kept >= args.samples:
                break

            # Scatter cursor so consecutive frames see different
            # backgrounds and different cursor regions.
            sdx, sdy = _safe_scatter(cursor, rng)
            await _send(mouse, sdx, sdy)
            await asyncio.sleep(0.25)

            # Capture the input frame — this is what the future model
            # will see at inference time.
            input_frame = await _capture_color(cap)
            grab_ts = time.time()

            # Label it via oscillation. The variance-centroid recovers
            # the cursor's position AT THE START of the oscillation,
            # which is exactly where the cursor was in `input_frame`.
            pos = await _locate_cursor(cap, mouse)
            if pos is None:
                missed += 1
                continue

            # Sanity: oscillation should land near the prior `cursor`
            # plus the scatter delta. If it's way off, something is
            # wrong — discard.
            expected = (cursor[0] + sdx * 0.0008, cursor[1] + sdy * 0.0008)
            dist = math.hypot(pos[0] - expected[0], pos[1] - expected[1])
            if dist > 0.20:
                missed += 1
                continue

            fname = f"{kept:05d}.png"
            cv2.imwrite(str(frames_dir / fname), input_frame)
            row = {
                "frame": f"frames/{fname}",
                "x_pct": float(pos[0]),
                "y_pct": float(pos[1]),
                "ts": grab_ts,
                "brightness": float(input_frame.mean()) / 255.0,
            }
            with labels_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            kept += 1
            cursor = pos

            if kept and kept % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "%d kept / %d missed (%.0f rows/min)",
                    kept, missed, kept / elapsed * 60,
                )
    except KeyboardInterrupt:
        logger.info("interrupted; %d rows written", kept)
    finally:
        try: await mouse.disconnect()
        except Exception: pass
        try: await cap.close()
        except Exception: pass

    elapsed = time.time() - t0
    print(
        f"\n=== cursor-detector collection summary ===\n"
        f"  kept:    {kept}\n"
        f"  missed:  {missed}\n"
        f"  elapsed: {elapsed:.1f}s\n"
        f"  rate:    {(kept / elapsed * 60 if elapsed else 0):.0f} rows/min\n"
        f"  out:     {out_dir}\n",
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=None)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument(
        "--out-root", type=str, default="data/ml/cursor_detector",
        help="Parent directory for the session output dir.",
    )
    ap.add_argument("--seed", type=int, default=-1)
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(_run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
