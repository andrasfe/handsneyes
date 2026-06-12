"""Live mouse-precision battery against the running Command Center.

Fires click_at at a set of aim points and — independently of the
homer's own telemetry — verifies each landing by template-matching
the target's stock cursor in a fresh post-click frame. The homer
grading itself is not evidence; the template match is.

Usage:
    .venv/bin/python scripts/precision_battery.py \
        [--cc http://localhost:8765] \
        [--points "0.25,0.60 0.65,0.30 ..."]

CAUTION: every point is a REAL click on the target machine. Choose
aim points on inert surfaces (desktop background, editor/terminal
panes) — never over buttons, dock icons, or settings toggles.

A point lands "unverified" when the cursor at the landing spot is
not the arrow (e.g. I-beam over a text area) — the homer's converged
residual in the cc logs is the only telemetry for those.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import urllib.request

import cv2

from handsneyes.core.vision.cursor_finder import (
    find_cursor_template_multiscale,
)
from handsneyes.platforms.macos.cursor_templates import (
    stock_cursor_templates,
)

DEFAULT_POINTS = "0.25,0.60 0.65,0.30 0.70,0.70 0.25,0.25 0.62,0.55 0.28,0.80"


def api(cc: str, path: str, body: dict | None = None, timeout: int = 90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        cc + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def latest_frame() -> str:
    files = sorted(
        glob.glob(os.path.expanduser(
            "~/.local/share/handsneyes/runs/manual/*.png",
        )),
        key=os.path.getmtime,
    )
    return files[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="http://localhost:8765")
    ap.add_argument("--points", default=DEFAULT_POINTS)
    args = ap.parse_args()

    points = [
        tuple(float(v) for v in p.split(","))
        for p in args.points.split()
    ]
    tpls = stock_cursor_templates()
    errs = []
    for i, (x, y) in enumerate(points):
        t0 = time.time()
        resp = api(args.cc, "/api/mouse/click_at", {
            "x_pct": x, "y_pct": y, "button": "left", "count": 1,
        })
        dt = time.time() - t0
        time.sleep(0.6)
        api(args.cc, "/api/snapshot", {}, timeout=30)
        time.sleep(0.5)
        img = cv2.imread(latest_frame())
        h, w = img.shape[:2]
        hit = find_cursor_template_multiscale(
            img, tpls, search_center_pct=(x, y), search_radius_pct=0.15,
        )
        if hit:
            ex = (hit.x_pct - x) * w
            ey = (hit.y_pct - y) * h
            errs.append(math.hypot(ex, ey))
            print(
                f"[{i}] aim=({x:.2f},{y:.2f}) t={dt:.1f}s "
                f"err=({ex:+.1f},{ey:+.1f})px "
                f"tpl={hit.template_name}@{hit.score:.2f} "
                f"api={resp.get('reason')}"
            )
        else:
            print(
                f"[{i}] aim=({x:.2f},{y:.2f}) t={dt:.1f}s "
                f"err=UNVERIFIED (non-arrow cursor?) "
                f"api={resp.get('reason')}"
            )
    if errs:
        s = sorted(errs)
        print(
            f"\nverified {len(errs)}/{len(points)}: "
            f"median={s[len(s) // 2]:.1f}px "
            f"mean={sum(errs) / len(errs):.1f}px max={max(errs):.1f}px "
            f"(frame {w}x{h})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
