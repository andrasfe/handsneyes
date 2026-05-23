#!/usr/bin/env python3
"""host_collect.py — run ON THE HOST (not the dev mac).

The dev-mac collector (explore_pointer_accel*.py) sees the cursor
through a webcam, which adds ±3-5px label noise per sample. That
noise floor capped every retrain experiment we tried.

This script runs ON THE TARGET MACHINE. It injects relative mouse
deltas through ``/dev/uinput`` so libinput sees them exactly as it
would a Bluetooth HID report from the Pi, then reads the cursor's
post-move pixel position via ``xdotool getmouselocation`` — both
sides of the (HID, pixel) pair are now pixel-exact, not webcam-
estimated.

Output schema matches the existing
``<run>/homer/<id>/history.jsonl`` so the dev-mac trainer can
pull the file directly (over scp / syncthing / nfs / whatever)
into ``data/ml/pointer_accel/`` and re-train without any builder
changes.

Throughput: ~1000-2000 rows/min (no oscillation overhead).

Deployment:
  # On the host (Ubuntu/GNOME):
  sudo apt install xdotool python3-evdev
  sudo usermod -aG input $USER   # then log out + back in
  # Or run with sudo if you'd rather not change groups.

  scp scripts/host_collect.py <host>:~/
  ssh <host>
  python3 host_collect.py --samples 5000 --out ~/host_collect.jsonl

  # Back on the dev mac:
  scp <host>:~/host_collect.jsonl <somewhere under runs-root>/host_collect/homer/<id>/history.jsonl
  python scripts/build_pointer_accel_dataset.py
  python scripts/train_pointer_accel.py --output data/ml/checkpoints/pointer_accel-host-v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

def _import_evdev():
    """Import evdev lazily so --help works on machines without it.

    Linux-only — fails clearly if not installed."""
    try:
        from evdev import UInput, ecodes
        return UInput, ecodes
    except ImportError:
        print(
            "ERROR: python-evdev not available. Install with:\n"
            "  sudo apt install python3-evdev   (or)\n"
            "  pip install evdev\n",
            file=sys.stderr,
        )
        sys.exit(1)


_XDOTOOL_LOC_RE = re.compile(r"x:(\d+)\s+y:(\d+)\s+screen:")


def _get_cursor() -> tuple[int, int] | None:
    """Return (x, y) cursor pixel position via xdotool. None on failure."""
    try:
        out = subprocess.check_output(
            ["xdotool", "getmouselocation"], timeout=2.0,
        ).decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return None
    m = _XDOTOOL_LOC_RE.search(out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _detect_screen_size() -> tuple[int, int]:
    """Return (width, height) via xdotool. Fallback to 1920x1080."""
    try:
        out = subprocess.check_output(
            ["xdotool", "getdisplaygeometry"], timeout=2.0,
        ).decode("utf-8", errors="replace").strip().split()
        return int(out[0]), int(out[1])
    except Exception:
        return 1920, 1080


def _open_uinput():
    """Create a virtual relative-mouse device through /dev/uinput.

    libinput treats this as a real pointer; the kernel's acceleration
    profile applies identically to events injected here and to events
    arriving from a Bluetooth HID gadget."""
    UInput, ecodes = _import_evdev()
    caps = {
        ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y, ecodes.REL_WHEEL],
        ecodes.EV_KEY: [
            ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE,
        ],
    }
    return UInput(caps, name="handsneyes-host-collect", version=0x1), ecodes


def _send_move(ui, ecodes, dx: int, dy: int) -> None:
    """Inject a single REL_X/REL_Y event pair, like one HID report.

    Real BT HID clamps each report to ±127 — we mirror that so the
    libinput curve sees the same per-event magnitudes. Large logical
    moves are split into multiple reports, matching the Pi's behaviour.
    """
    while abs(dx) > 127 or abs(dy) > 127:
        sx = max(-127, min(127, dx))
        sy = max(-127, min(127, dy))
        ui.write(ecodes.EV_REL, ecodes.REL_X, sx)
        ui.write(ecodes.EV_REL, ecodes.REL_Y, sy)
        ui.syn()
        dx -= sx
        dy -= sy
        time.sleep(0.005)
    if dx or dy:
        ui.write(ecodes.EV_REL, ecodes.REL_X, dx)
        ui.write(ecodes.EV_REL, ecodes.REL_Y, dy)
        ui.syn()


def _sample_hid(
    cursor_pct: tuple[float, float], *,
    mag_min: int, mag_max: int, margin: float, rng: random.Random,
) -> tuple[int, int]:
    log_mag = rng.uniform(math.log(mag_min), math.log(mag_max))
    mag = math.exp(log_mag)
    theta = rng.uniform(0.0, 2 * math.pi)
    dx = int(round(mag * math.cos(theta)))
    dy = int(round(mag * math.sin(theta)))
    # Rough ratio (will be replaced by the model once we train one);
    # only used to keep predicted destinations inside the safe band.
    cx, cy = cursor_pct
    rough = 0.0008
    if cx + dx * rough < margin or cx + dx * rough > 1 - margin:
        dx = -dx
    if cy + dy * rough < margin or cy + dy * rough > 1 - margin:
        dy = -dy
    return dx, dy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--mag-min", type=int, default=2)
    ap.add_argument("--mag-max", type=int, default=120)
    ap.add_argument(
        "--settle-ms", type=int, default=15,
        help="Sleep (ms) between injection and post-position read. "
             "libinput needs ~5-10ms to deliver the event to the "
             "compositor and update the cursor position. 15ms is "
             "comfortably above the floor with no perceptible cost.",
    )
    ap.add_argument(
        "--edge-margin", type=float, default=0.05,
        help="Sign-flip if predicted destination lands within this "
             "fraction of any edge.",
    )
    ap.add_argument(
        "--out", type=str, default="~/host_collect.jsonl",
        help="Output path. Schema matches the dev-mac's homer "
             "history.jsonl so build_pointer_accel_dataset.py picks "
             "it up directly.",
    )
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed if args.seed >= 0 else None)

    if os.geteuid() != 0 and not os.access("/dev/uinput", os.W_OK):
        print(
            "WARN: /dev/uinput not writable by this user. Either "
            "run with sudo, or `sudo usermod -aG input $USER` + "
            "logout/login.",
            file=sys.stderr,
        )

    sw, sh = _detect_screen_size()
    print(f"screen: {sw}x{sh}")

    init = _get_cursor()
    if init is None:
        print("ERROR: xdotool getmouselocation failed. Install xdotool.",
              file=sys.stderr)
        return 2
    print(f"initial cursor: {init}")

    ui, ecodes = _open_uinput()
    time.sleep(0.5)  # let uinput device register with libinput

    kept = 0
    dropped = 0
    settle_s = args.settle_ms / 1000.0
    t0 = time.time()

    try:
        with out.open("a", encoding="utf-8") as f:
            for step_idx in range(args.samples * 2):  # headroom for drops
                if kept >= args.samples:
                    break
                pre = _get_cursor()
                if pre is None:
                    dropped += 1
                    continue
                cursor_pct = (pre[0] / sw, pre[1] / sh)
                hid_dx, hid_dy = _sample_hid(
                    cursor_pct,
                    mag_min=args.mag_min, mag_max=args.mag_max,
                    margin=args.edge_margin, rng=rng,
                )
                _send_move(ui, ecodes, hid_dx, hid_dy)
                time.sleep(settle_s)
                post = _get_cursor()
                if post is None:
                    dropped += 1
                    continue
                mdx_pct = (post[0] - pre[0]) / sw
                mdy_pct = (post[1] - pre[1]) / sh
                # Sanity gate matches the dev-mac builder so kept rows
                # are guaranteed-trainable.
                ok = True
                for h, m in ((hid_dx, mdx_pct), (hid_dy, mdy_pct)):
                    if abs(h) < 3:
                        continue
                    ratio = abs(m / h) if h else 0
                    if ratio < 3e-4 or ratio > 8e-3:
                        ok = False
                        break
                if not ok:
                    dropped += 1
                    continue
                row = {
                    "cursor_img": list(cursor_pct),
                    "target_img": [
                        cursor_pct[0] + hid_dx * 0.0008,
                        cursor_pct[1] + hid_dy * 0.0008,
                    ],
                    "hid_dx": int(hid_dx),
                    "hid_dy": int(hid_dy),
                    "measured_dx_pct": float(mdx_pct),
                    "measured_dy_pct": float(mdy_pct),
                    "ratio_x": (
                        abs(mdx_pct / hid_dx) if abs(hid_dx) >= 3 else None
                    ),
                    "ratio_y": (
                        abs(mdy_pct / hid_dy) if abs(hid_dy) >= 3 else None
                    ),
                    "note": "host_collect_uinput",
                    "ts": time.time(),
                    "step_idx": step_idx,
                }
                f.write(json.dumps(row) + "\n")
                kept += 1
                if kept % 250 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  {kept} kept / {dropped} dropped "
                        f"({kept / elapsed * 60:.0f} rows/min)",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\ninterrupted — flushed", kept, "rows")
    finally:
        ui.close()

    elapsed = time.time() - t0
    print(
        f"\n=== host_collect summary ===\n"
        f"  kept:     {kept}\n"
        f"  dropped:  {dropped}\n"
        f"  elapsed:  {elapsed:.1f}s\n"
        f"  rate:     {kept / elapsed * 60:.0f} rows/min\n"
        f"  out:      {out}\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
