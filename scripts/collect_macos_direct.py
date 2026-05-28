#!/usr/bin/env python3
"""collect_macos_direct.py — clean training data for the macOS
pointer_accel model, bypassing the homer.

The homer's vision-based cursor detection doesn't work on macOS self-
capture because the OS doesn't composite the cursor into the
framebuffer that `screencapture` / `ImageGrab` see. Every history
row ends up with `measured_dx_pct: null` and the dataset builder
drops it.

But macOS does expose the cursor location through Quartz —
`CGEventGetLocation` returns pixel-precise coordinates with zero
latency. So for macOS we can skip the CV entirely:

  1. Read cursor pos via Quartz.
  2. Send a known HID delta (dx, dy) via the cc's /api/mouse/move.
  3. Wait briefly for the move to settle.
  4. Read cursor pos again via Quartz.
  5. Record (hid, measured_pixel_delta, cursor_before_position).
  6. Repeat with varied amplitudes + start positions.

Output is a JSONL file at `<output_dir>/homer/macos_direct_<ts>/
history.jsonl`, schema-compatible with the homer's records so the
existing build_pointer_accel_dataset.py / train_pointer_accel.py
pipeline ingests it without changes.

Usage:
  python scripts/collect_macos_direct.py [--samples 500] [--base URL]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import Quartz
import httpx


def cursor_xy() -> tuple[float, float]:
    e = Quartz.CGEventCreate(None)
    p = Quartz.CGEventGetLocation(e)
    return float(p.x), float(p.y)


def screen_dims() -> tuple[int, int]:
    m = Quartz.CGMainDisplayID()
    return int(Quartz.CGDisplayPixelsWide(m)), int(Quartz.CGDisplayPixelsHigh(m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path.home() / ".local/share/handsneyes/runs",
    )
    ap.add_argument("--settle-ms", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sw, sh = screen_dims()
    print(f"screen: {sw}x{sh}")

    # Carve out a session dir. Match the homer's layout so existing
    # tools find it.
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    session = args.output_dir / ts / "homer" / f"macos_direct_{ts}"
    session.mkdir(parents=True, exist_ok=True)
    out = session / "history.jsonl"
    print(f"writing → {out}")

    # HID amplitude grid. Spread heavily across the full range so
    # the model sees enough large-amplitude examples to extrapolate
    # for slam-to-target jumps. v1 was biased toward small
    # amplitudes and undershot on large-residual targets.
    amplitudes = [
        2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 110, 120, 127,
        # Repeat the big ones so they appear ~3x more often.
        80, 90, 100, 110, 120, 127,
        70, 80, 100, 110, 120, 127,
    ]

    client = httpx.Client(base_url=args.base, timeout=10.0)
    n_written = 0
    n_skipped = 0

    # Anchor positions across the screen. Each anchor is visited
    # multiple times with varied HID deltas to populate the (cursor,
    # hid) joint distribution.
    # Anchors spread wider including extremes near 0.10 / 0.90 so
    # the model has data for cursor positions at the edges (v1
    # consistently stuck around x=0.6 for x=0.80 targets — too few
    # right-half samples).
    anchors = [
        (0.10, 0.25), (0.30, 0.25), (0.50, 0.25), (0.70, 0.25), (0.90, 0.25),
        (0.10, 0.50), (0.30, 0.50), (0.50, 0.50), (0.70, 0.50), (0.90, 0.50),
        (0.10, 0.70), (0.30, 0.70), (0.50, 0.70), (0.70, 0.70), (0.90, 0.70),
    ]

    target_idx = 0
    samples_per_anchor = max(1, args.samples // len(anchors))
    print(f"~{samples_per_anchor} samples per anchor, {len(anchors)} anchors")

    try:
        while n_written < args.samples:
            anchor_x_pct, anchor_y_pct = anchors[
                target_idx % len(anchors)
            ]
            target_idx += 1

            # 1. Slam to corner so the position is known.
            for _ in range(8):
                client.post(
                    "/api/mouse/move",
                    json={"dx": -127, "dy": -127},
                )
            time.sleep(0.25)

            # 2. Move to the anchor area using a rough open-loop ratio.
            # macOS pointer-acceleration ≈ 0.0008 pct-per-hid at small
            # velocities; coarser at large. Just get within ~30% of
            # the anchor; the per-sample HID is what we're really
            # measuring.
            target_x_px = anchor_x_pct * sw
            target_y_px = anchor_y_pct * sh
            for _ in range(2):
                cx, cy = cursor_xy()
                ratio = 0.0008  # rough seed
                rem_dx_px = target_x_px - cx
                rem_dy_px = target_y_px - cy
                hid_dx = max(-127, min(127, int(
                    rem_dx_px / sw / ratio
                )))
                hid_dy = max(-127, min(127, int(
                    rem_dy_px / sh / ratio
                )))
                if abs(hid_dx) < 2 and abs(hid_dy) < 2:
                    break
                client.post(
                    "/api/mouse/move",
                    json={"dx": hid_dx, "dy": hid_dy},
                )
                time.sleep(args.settle_ms / 1000.0)

            # 3. Now take N samples at varied HID amplitudes.
            for _ in range(samples_per_anchor):
                if n_written >= args.samples:
                    break
                amp = rng.choice(amplitudes)
                # Random signs + axis weighting so we cover the
                # full HID space, not just monotone moves.
                hid_dx = amp * rng.choice([-1, 1])
                hid_dy = amp * rng.choice([-1, 1])
                # Sometimes zero out one axis for axis-aligned moves.
                if rng.random() < 0.3:
                    if rng.random() < 0.5:
                        hid_dx = 0
                    else:
                        hid_dy = 0
                # Skip the (0,0) case the dataset filter rejects.
                if hid_dx == 0 and hid_dy == 0:
                    continue

                before_x, before_y = cursor_xy()
                # Skip if the cursor is at the screen edge — the move
                # would clip and the measured delta would be wrong.
                if (
                    before_x < 5 or before_x > sw - 5
                    or before_y < 5 or before_y > sh - 5
                ):
                    # Re-centre via the anchor pass next iteration.
                    n_skipped += 1
                    break

                client.post(
                    "/api/mouse/move",
                    json={"dx": hid_dx, "dy": hid_dy},
                )
                time.sleep(args.settle_ms / 1000.0)
                after_x, after_y = cursor_xy()

                measured_dx_pct = (after_x - before_x) / sw
                measured_dy_pct = (after_y - before_y) / sh

                # Reject samples where the cursor didn't move at all
                # (HID got clipped at the edge, or pointer-accel
                # absorbed it).
                if abs(measured_dx_pct) < 1e-5 and abs(measured_dy_pct) < 1e-5:
                    n_skipped += 1
                    continue

                row = {
                    "cursor_img": [before_x / sw, before_y / sh],
                    "target_img": [
                        anchor_x_pct, anchor_y_pct,
                    ],
                    "residual_pct": 0.0,
                    "hid_dx": hid_dx,
                    "hid_dy": hid_dy,
                    "measured_dx_pct": measured_dx_pct,
                    "measured_dy_pct": measured_dy_pct,
                    "ratio_x": (
                        abs(measured_dx_pct / hid_dx)
                        if hid_dx else 0.0
                    ),
                    "ratio_y": (
                        abs(measured_dy_pct / hid_dy)
                        if hid_dy else 0.0
                    ),
                    "note": "macos_direct",
                    "ts": time.time(),
                    "platform": "macos",
                }
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                n_written += 1

                if n_written % 50 == 0:
                    print(
                        f"  {n_written}/{args.samples} "
                        f"(skipped {n_skipped})"
                    )

    finally:
        client.close()

    print(f"done: wrote {n_written} rows, skipped {n_skipped}, → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
