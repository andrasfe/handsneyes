#!/usr/bin/env python3
"""explore_pointer_accel_v2.py — target-driven exploration.

Phase-1's v1 sampled random HID deltas and recorded whatever pixel
deltas the libinput acceleration curve produced. That trained the
model on the distribution the *curve* emits, but the homer queries
it with the distribution the *targets* demand — heavily small,
neighbourhood-scale deltas.

v2 inverts the loop: sample TARGET pixel deltas in the cursor's
neighbourhood (log-uniform, weighted toward small), use the current
model to predict an HID, send it, and record the error. The model's
own predictions become the training signal — active learning,
focused exactly where the model is currently weak.

Per-sample loop (~2-3s wall time):
  1. Detect cursor C via oscillation-variance.
  2. Sample a target offset (Δx, Δy) — log-uniform pixel magnitude
     in [--target-min, --target-max] pct, direction uniform on the
     unit circle. Sign-flip if T would land outside the safe band.
  3. Ask current_model.inverse(target_dx=Δx, target_dy=Δy, cursor=C)
     for HID. Bootstrap with the shipped yaru-v4 checkpoint.
  4. Send HID via Pi. Wait --settle.
  5. Detect cursor C'. Record:
       cursor_img = C
       target_img = C + (Δx, Δy)            ← what we aimed for
       hid_dx, hid_dy                        ← what the model said
       measured_dx_pct, measured_dy_pct = C' - C   ← what we got
  6. Every --scatter-every samples, pick a far destination on
     screen and walk the cursor there (using the model). This
     diffuses the (cursor_x, cursor_y) distribution naturally —
     no random bursts, no synthetic resets.

Row schema matches v1 → matches the homer's history.jsonl →
build_pointer_accel_dataset.py picks it up unchanged.

Each --batch-size samples gets a fresh trajectory_id so the
dataset builder's 80/10/10 split produces real val/test rows
instead of a single-trajectory train-only blob.

Usage::

    python scripts/explore_pointer_accel_v2.py --samples 50
    python scripts/explore_pointer_accel_v2.py --samples 500 \\
        --target-min 0.005 --target-max 0.30 --batch-size 25
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from handsneyes.core.capture.webcam import WebcamCapture  # noqa: E402
from handsneyes.core.vision.cursor_finder import (  # noqa: E402
    find_cursor_by_variance,
)
from handsneyes.core.vision.pointer_accel import (  # noqa: E402
    PointerAccelModel,
)
from handsneyes.io.mouse.backends.http import HttpMouseOutput  # noqa: E402
from handsneyes.platforms import load_adapter  # noqa: E402
from handsneyes.targets import TargetRegistry  # noqa: E402

logger = logging.getLogger("explore_v2")

_ASLEEP_BRIGHTNESS = 0.06

# Symmetric jiggle for oscillation-variance detection. Mirrors the
# homer's pattern so labels are comparable to runtime measurements.
_OSCILLATION = [(20, 0), (-40, 0), (40, 0), (0, 20), (0, -40), (0, 40)]


# ── HID + capture helpers ──────────────────────────────────────────


async def _capture_gray(cap: WebcamCapture) -> np.ndarray:
    frame = await cap.capture_frame()
    img = frame.image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


async def _send(mouse: HttpMouseOutput, dx: int, dy: int) -> None:
    """Split HID deltas > 120 into chunks (Pi clamps to ±127)."""
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
    frames: list[np.ndarray] = [await _capture_gray(cap)]
    for dx, dy in _OSCILLATION:
        await _send(mouse, dx, dy)
        await asyncio.sleep(0.10)
        frames.append(await _capture_gray(cap))
    hit = find_cursor_by_variance(frames)
    if debug_dir is not None and label:
        try:
            arr = np.stack([f.astype(np.float32) for f in frames], axis=0)
            var = arr.std(axis=0)
            vmax = float(var.max()) if var.size else 1.0
            vis = (var / max(vmax, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{label}_frame0.png"), frames[0])
            cv2.imwrite(str(debug_dir / f"{label}_variance.png"), vis)
        except Exception:
            pass
    return hit


def _looks_like_test_pattern(img: np.ndarray) -> bool:
    """SMPTE-style color bars / grayscale step wedges have vertical
    bands of constant intensity → vertical gradient ≈ 0 everywhere.
    A real captured screen has text, icons, and window edges → much
    higher vertical gradient. Threshold chosen with ~3× margin
    around observed test-pattern values."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return False
    vert_grad = float(np.abs(np.diff(gray.astype(np.int16), axis=0)).mean())
    return vert_grad < 1.5


async def _wake_target(
    cap: WebcamCapture, mouse: HttpMouseOutput,
) -> tuple[bool, float]:
    frame = await cap.capture_frame()
    img = frame.image
    if img is None:
        return False, 0.0
    if _looks_like_test_pattern(img):
        logger.error(
            "webcam is returning the SMPTE-bar / grayscale-wedge test "
            "pattern. The camera isn't seeing the target. Check: "
            "(1) another app holding the camera (Photo Booth, Slack, "
            "Zoom, Meet, FaceTime); (2) macOS Settings → Privacy & "
            "Security → Camera permission for your terminal/Python; "
            "(3) USB cable / cam unplugged.",
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


# ── target-delta sampler ───────────────────────────────────────────


def _sample_target_delta(
    cursor: tuple[float, float],
    *,
    pct_min: float, pct_max: float,
    margin: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Log-uniform magnitude (in pct of screen) × uniform direction.

    Sign-flips axes if the predicted destination would land within
    ``margin`` of any edge. Margin keeps the cursor in the interior
    so subsequent samples have headroom in every direction.
    """
    log_mag = rng.uniform(math.log(pct_min), math.log(pct_max))
    mag = math.exp(log_mag)
    theta = rng.uniform(0.0, 2 * math.pi)
    dx = mag * math.cos(theta)
    dy = mag * math.sin(theta)
    cx, cy = cursor
    if cx + dx < margin or cx + dx > 1 - margin:
        dx = -dx
    if cy + dy < margin or cy + dy > 1 - margin:
        dy = -dy
    return dx, dy


# ── main session ───────────────────────────────────────────────────


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
        "target=%s pi=%s cam=%d platform=%s",
        target.name, target.pi_url, target.camera_index, target.platform,
    )

    # Load the platform's pointer_accel checkpoint as the bootstrap
    # model. v2 needs a working seed model from sample 1 — the
    # shipped yaru-v4 is the natural choice.
    adapter = load_adapter(target.platform)
    ckpt = adapter.pointer_accel_checkpoint() if hasattr(
        adapter, "pointer_accel_checkpoint",
    ) else None
    if ckpt is None or not ckpt.exists():
        logger.error(
            "no pointer_accel checkpoint for platform %r — v2 needs a "
            "bootstrap model. Train a v1 first or ship a checkpoint.",
            target.platform,
        )
        return 5
    model = PointerAccelModel(ckpt)
    logger.info("bootstrap model: %s", ckpt)

    # Output dirs.
    runs_root = (
        Path(args.runs_root) if args.runs_root
        else Path.home() / ".local/share/handsneyes/runs"
    )
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = runs_root / f"explore_v2_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = session_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    logger.info("session dir: %s", session_dir)

    # I/O.
    cap = WebcamCapture(device_index=target.camera_index)
    await cap.open()
    mouse = HttpMouseOutput(
        base_url=target.pi_url, transport=target.transport,
    )
    await mouse.connect()

    kept = 0
    dropped_detect = 0
    dropped_sanity = 0
    total_err_pct = 0.0
    t0 = time.time()

    # Per-batch state.
    batch_idx = 0
    batch_dir: Path | None = None
    history_path: Path | None = None
    rows_in_batch = 0

    def _open_new_batch() -> tuple[Path, Path]:
        nonlocal batch_idx, batch_dir, history_path, rows_in_batch
        homer_id = f"explorev2-{batch_idx:03d}-{uuid.uuid4().hex[:6]}"
        bd = session_dir / "homer" / homer_id
        bd.mkdir(parents=True, exist_ok=True)
        hp = bd / "history.jsonl"
        batch_idx += 1
        rows_in_batch = 0
        return bd, hp

    batch_dir, history_path = _open_new_batch()

    try:
        awake, brightness = await _wake_target(cap, mouse)
        if not awake:
            logger.error(
                "screen dark after wake (brightness=%.3f) — aborting",
                brightness,
            )
            return 2

        cursor = await _locate_cursor(
            cap, mouse, debug_dir=debug_dir, label="preflight",
        )
        if cursor is None:
            logger.error(
                "initial cursor detect failed — see %s", debug_dir,
            )
            return 2

        step_idx = 0
        while kept < args.samples:
            target_dx, target_dy = _sample_target_delta(
                cursor,
                pct_min=args.target_min, pct_max=args.target_max,
                margin=args.edge_margin, rng=rng,
            )

            # Ask the bootstrap model for HID. This is the active-
            # learning core: the model picks "what to try", we
            # measure the error.
            hid_dx, hid_dy = model.inverse(
                target_dx_pct=target_dx, target_dy_pct=target_dy,
                cursor_x_pct=cursor[0], cursor_y_pct=cursor[1],
            )

            try:
                await _send(mouse, hid_dx, hid_dy)
            except Exception as e:  # noqa: BLE001
                logger.warning("send failed: %s", e)
                await asyncio.sleep(2.0)
                continue
            await asyncio.sleep(args.settle)

            cursor_after = await _locate_cursor(cap, mouse)
            if cursor_after is None:
                dropped_detect += 1
                continue

            mdx = cursor_after[0] - cursor[0]
            mdy = cursor_after[1] - cursor[1]

            # Sanity: per-axis ratio sane + delta within order of
            # magnitude of what was asked. The dataset builder will
            # apply its own [3e-4, 8e-3] gate downstream; we use
            # the same here so kept ≈ trainable.
            ok = True
            for h, m in ((hid_dx, mdx), (hid_dy, mdy)):
                if abs(h) < 3:
                    continue
                ratio = abs(m / h)
                if ratio < 3e-4 or ratio > 8e-3:
                    ok = False
                    break
            # Opposite-direction motion > 1% screen = misdetection.
            if abs(mdx) > 0.01 and target_dx * mdx < 0:
                ok = False
            if abs(mdy) > 0.01 and target_dy * mdy < 0:
                ok = False

            # Error w.r.t. the *target*. This is the metric the
            # model is actually trying to minimise; useful as a
            # live quality signal.
            err = math.hypot(target_dx - mdx, target_dy - mdy)

            row = {
                "cursor_img": [cursor[0], cursor[1]],
                "target_img": [cursor[0] + target_dx, cursor[1] + target_dy],
                "hid_dx": int(hid_dx), "hid_dy": int(hid_dy),
                "measured_dx_pct": float(mdx), "measured_dy_pct": float(mdy),
                "ratio_x": (abs(mdx / hid_dx) if abs(hid_dx) >= 3 else None),
                "ratio_y": (abs(mdy / hid_dy) if abs(hid_dy) >= 3 else None),
                "target_dx_pct": float(target_dx),
                "target_dy_pct": float(target_dy),
                "residual_pct": float(err),
                "note": "explore_v2_target_driven",
                "ts": time.time(),
                "step_idx": step_idx,
                "sanity_ok": bool(ok),
            }
            step_idx += 1

            if ok:
                assert history_path is not None
                with history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                kept += 1
                rows_in_batch += 1
                total_err_pct += err

                # Rotate to a new trajectory_id every --batch-size.
                if rows_in_batch >= args.batch_size:
                    batch_dir, history_path = _open_new_batch()
            else:
                dropped_sanity += 1

            cursor = cursor_after

            if kept and kept % 25 == 0 and ok:
                elapsed = time.time() - t0
                mean_err = total_err_pct / kept
                logger.info(
                    "%d kept / %d miss / %d sanity (%.1f rows/min, "
                    "mean target-error %.3f pct)",
                    kept, dropped_detect, dropped_sanity,
                    kept / elapsed * 60, mean_err,
                )
    except KeyboardInterrupt:
        logger.info("interrupted; flushed %d rows", kept)
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
    mean_err = total_err_pct / max(kept, 1)
    print(
        f"\n=== explore_v2 summary ===\n"
        f"  kept:              {kept}\n"
        f"  dropped (detect):  {dropped_detect}\n"
        f"  dropped (sanity):  {dropped_sanity}\n"
        f"  trajectories:      {batch_idx}\n"
        f"  elapsed:           {elapsed:.1f}s\n"
        f"  rate:              {(kept / elapsed * 60 if elapsed else 0):.1f} rows/min\n"
        f"  mean target-error: {mean_err:.4f} pct (lower = bootstrap "
        f"model already good at this regime)\n"
        f"  session dir:       {session_dir}\n",
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=None)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument(
        "--target-min", type=float, default=0.005,
        help="Smallest target-delta magnitude (pct of screen). 0.005 = "
             "~5px on a 1080-tall screen.",
    )
    ap.add_argument(
        "--target-max", type=float, default=0.30,
        help="Largest target-delta magnitude (pct of screen). 0.30 = "
             "~30%% of screen — covers first-iteration slams.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=25,
        help="Rotate to a fresh trajectory_id every N rows so the "
             "dataset builder's 80/10/10 split has multiple "
             "trajectories to choose from.",
    )
    ap.add_argument(
        "--settle", type=float, default=0.20,
        help="Seconds after the test HID before re-detecting.",
    )
    ap.add_argument(
        "--edge-margin", type=float, default=0.08,
        help="Sign-flip if the target would land within this fraction "
             "of any edge.",
    )
    ap.add_argument(
        "--runs-root", type=str, default=None,
        help="Override the runs-root output dir.",
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
    # Silence httpx noise — one INFO per HTTP request floods the log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(_run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
