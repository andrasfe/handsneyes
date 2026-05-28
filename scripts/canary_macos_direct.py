#!/usr/bin/env python3
"""canary_macos_direct.py — test the trained macOS pointer_accel model
without the homer's CV loop.

The homer's cursor-measurement step uses oscillation-variance /
frame-diff against the captured frame. On macOS the captured frame
doesn't include the cursor (it's in a separate hardware layer), so
the loop can't tell where the cursor went and never converges. This
canary runs the same model+HID logic but reads cursor position via
``Quartz.CGEventGetLocation`` — pixel-precise, zero latency.

For each target point: slam to a corner, then iteratively predict
HID via the model + send + measure via Quartz until within
tolerance. Reports steps per click.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import Quartz
import httpx
import numpy as np


def cursor_pct(sw: int, sh: int) -> tuple[float, float]:
    e = Quartz.CGEventCreate(None)
    p = Quartz.CGEventGetLocation(e)
    return float(p.x) / sw, float(p.y) / sh


def screen_dims() -> tuple[int, int]:
    m = Quartz.CGMainDisplayID()
    return (
        int(Quartz.CGDisplayPixelsWide(m)),
        int(Quartz.CGDisplayPixelsHigh(m)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", type=Path,
        default=Path("data/ml/checkpoints/pointer_accel-macos-v1"),
    )
    ap.add_argument(
        "--base", default="http://127.0.0.1:8765",
    )
    ap.add_argument("--max-iters", type=int, default=15)
    ap.add_argument(
        "--tol-pct", type=float, default=0.008,
        help="Per-axis convergence tolerance in pct.",
    )
    args = ap.parse_args()

    from handsneyes.core.vision.pointer_accel import PointerAccelModel
    model = PointerAccelModel(args.checkpoint)
    print(f"loaded model from {args.checkpoint}")
    print(f"  hidden={model.config.hidden}, direction={model.config.direction}")

    sw, sh = screen_dims()
    print(f"screen: {sw}x{sh}")
    client = httpx.Client(base_url=args.base, timeout=10.0)

    targets = [
        (0.20, 0.25), (0.50, 0.25), (0.80, 0.25),
        (0.20, 0.50), (0.80, 0.50),
        (0.20, 0.70), (0.50, 0.70), (0.80, 0.70),
    ]

    step_counts = []
    failures = []
    for tx, ty in targets:
        # Slam to top-left corner so we start from a known place.
        for _ in range(8):
            client.post("/api/mouse/move", json={"dx": -127, "dy": -127})
        time.sleep(0.3)

        for it in range(args.max_iters):
            cx, cy = cursor_pct(sw, sh)
            dx_pct = tx - cx
            dy_pct = ty - cy
            if abs(dx_pct) < args.tol_pct and abs(dy_pct) < args.tol_pct:
                step_counts.append(it)
                print(
                    f"({tx:.2f},{ty:.2f}): converged in {it} steps "
                    f"(final cursor: {cx:.3f},{cy:.3f}, residual: "
                    f"{abs(dx_pct):.4f},{abs(dy_pct):.4f})"
                )
                break

            # Hybrid: when the residual is large (way outside the
            # per-step amplitudes the model saw in training), send a
            # crude open-loop burst in the right direction. Only ask
            # the model to refine once we're close.
            #
            # macOS pointer-accel caps hid=127 at ≈0.036 pct/step. A
            # single-shot coarse seed therefore cannot cross large
            # distances within the iter budget. Burst multiple HID
            # sends within ONE iter when the residual exceeds what a
            # single +127 can cover, so the iter budget is spent on
            # refinement rather than slow open-loop crawl.
            BIG_DELTA = 0.05
            PER_HID_PCT = 0.036  # ≈ what hid=127 produces on macOS
            if abs(dx_pct) > BIG_DELTA or abs(dy_pct) > BIG_DELTA:
                # Coarse seed: ratio ≈ 0.0003 pct/hid → hid ≈ delta/0.0003
                sign_x = 1 if dx_pct >= 0 else -1
                sign_y = 1 if dy_pct >= 0 else -1
                bursts_x = max(1, int(abs(dx_pct) / PER_HID_PCT))
                bursts_y = max(1, int(abs(dy_pct) / PER_HID_PCT))
                bursts = max(bursts_x, bursts_y)
                # Don't blow past target — leave the final fraction for
                # model refinement.
                bursts = min(bursts, 6)
                for _ in range(bursts):
                    hid_dx_b = sign_x * 127 if abs(dx_pct) > 0.018 else 0
                    hid_dy_b = sign_y * 127 if abs(dy_pct) > 0.018 else 0
                    if hid_dx_b == 0 and hid_dy_b == 0:
                        break
                    client.post(
                        "/api/mouse/move",
                        json={"dx": hid_dx_b, "dy": hid_dy_b},
                    )
                    time.sleep(0.03)
                    cx_b, cy_b = cursor_pct(sw, sh)
                    dx_pct = tx - cx_b
                    dy_pct = ty - cy_b
                    if abs(dx_pct) < BIG_DELTA and abs(dy_pct) < BIG_DELTA:
                        break
                time.sleep(0.10)
                continue
            hid_dx, hid_dy = model.inverse(
                dx_pct, dy_pct, cx, cy,
            )
            client.post(
                "/api/mouse/move",
                json={"dx": int(hid_dx), "dy": int(hid_dy)},
            )
            time.sleep(0.10)
        else:
            cx, cy = cursor_pct(sw, sh)
            step_counts.append(args.max_iters)
            failures.append((tx, ty, cx, cy))
            print(
                f"({tx:.2f},{ty:.2f}): max_iters; final ({cx:.3f},{cy:.3f}) "
                f"residual ({abs(tx-cx):.3f},{abs(ty-cy):.3f})"
            )

    client.close()
    if step_counts:
        a = np.array(step_counts)
        print()
        print(f"=== summary ===")
        print(f"converged: {(a < args.max_iters).sum()}/{len(a)}")
        print(f"steps: mean={a.mean():.1f} median={np.median(a):.1f} "
              f"min={a.min()} max={a.max()}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
