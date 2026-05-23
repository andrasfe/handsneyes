#!/usr/bin/env python3
"""canary_pointer_accel.py — 8-point click_at benchmark.

Fires ``click_at`` at a fixed grid of 8 points (corners + edges +
centre) and records the homer's ``steps`` for each. Used to compare
pointer_accel checkpoints — install a new model, run this, compare
median/mean/max against the previous run.

Defaults assume the cc is running at http://localhost:8765.

Usage::

    python scripts/canary_pointer_accel.py
    python scripts/canary_pointer_accel.py --label v5 --out /tmp/canary_v5.json
    python scripts/canary_pointer_accel.py --label v4 --out /tmp/canary_v4.json
    python scripts/canary_pointer_accel.py --compare /tmp/canary_v4.json,/tmp/canary_v5.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

# 8 fixed canary positions — corners + edge midpoints + centre.
# Matches the set referenced in CLAUDE.md.
CANARY_POINTS: list[tuple[float, float]] = [
    (0.15, 0.15),  # TL
    (0.50, 0.15),  # T
    (0.85, 0.15),  # TR
    (0.15, 0.50),  # L
    (0.85, 0.50),  # R
    (0.15, 0.85),  # BL
    (0.50, 0.85),  # B
    (0.85, 0.85),  # BR
]


def _hit_one(client: httpx.Client, x: float, y: float) -> dict:
    t0 = time.time()
    r = client.post("/api/mouse/click_at", json={"x_pct": x, "y_pct": y})
    elapsed = time.time() - t0
    if r.status_code != 200:
        return {
            "x": x, "y": y, "ok": False,
            "steps": None, "elapsed_s": elapsed,
            "reason": f"HTTP {r.status_code}: {r.text[:120]}",
        }
    body = r.json()
    return {
        "x": x, "y": y,
        "ok": bool(body.get("ok")),
        "steps": body.get("steps"),
        "elapsed_s": elapsed,
        "reason": body.get("reason", ""),
    }


def _print_summary(label: str, results: list[dict]) -> dict:
    steps = [r["steps"] for r in results if r["ok"] and r["steps"] is not None]
    fails = [r for r in results if not r["ok"]]
    print(f"\n=== canary {label} ===")
    for r in results:
        flag = "✓" if r["ok"] else "✗"
        s = f"{r['steps']:>2d}" if r["steps"] is not None else "--"
        print(
            f"  {flag} ({r['x']:.2f},{r['y']:.2f}) "
            f"steps={s} elapsed={r['elapsed_s']:.1f}s  {r['reason'][:60]}"
        )
    if steps:
        summary = {
            "label": label,
            "n_ok": len(steps),
            "n_fail": len(fails),
            "median_steps": statistics.median(steps),
            "mean_steps": statistics.mean(steps),
            "min_steps": min(steps),
            "max_steps": max(steps),
            "all_steps": steps,
        }
        print(
            f"  → median={summary['median_steps']:.1f} "
            f"mean={summary['mean_steps']:.2f} "
            f"min={summary['min_steps']} max={summary['max_steps']} "
            f"({summary['n_ok']}/{len(results)} ok)"
        )
    else:
        summary = {
            "label": label,
            "n_ok": 0, "n_fail": len(fails),
            "median_steps": None, "mean_steps": None,
            "min_steps": None, "max_steps": None, "all_steps": [],
        }
        print("  → all clicks failed")
    return summary


def _compare(paths: list[Path]) -> int:
    rows: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    print(f"\n=== compare ({len(rows)} runs) ===")
    print(f"  {'label':<14s} {'n_ok':>4s} {'median':>7s} {'mean':>6s} "
          f"{'min':>4s} {'max':>4s}")
    for r in rows:
        med = f"{r['median_steps']:.1f}" if r['median_steps'] is not None else "  --"
        mean = f"{r['mean_steps']:.2f}" if r['mean_steps'] is not None else "  --"
        mn = f"{r['min_steps']}" if r['min_steps'] is not None else " --"
        mx = f"{r['max_steps']}" if r['max_steps'] is not None else " --"
        print(
            f"  {r['label']:<14s} {r['n_ok']:>4d} {med:>7s} {mean:>6s} "
            f"{mn:>4s} {mx:>4s}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cc-url", default="http://localhost:8765",
        help="Command Center base URL.",
    )
    ap.add_argument(
        "--label", default=None,
        help="Run label (e.g. 'v4', 'v5-active-learning'). Defaults to "
             "the current pointer_accel checkpoint dirname.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Write summary JSON here. Useful for later --compare.",
    )
    ap.add_argument(
        "--timeout", type=float, default=120.0,
        help="HTTP timeout per click_at (seconds).",
    )
    ap.add_argument(
        "--compare", default=None,
        help="Comma-separated list of summary JSON paths to compare. "
             "Skips collection.",
    )
    args = ap.parse_args()

    if args.compare:
        paths = [Path(p.strip()) for p in args.compare.split(",") if p.strip()]
        return _compare(paths)

    if args.label is None:
        # Auto-label from shipped checkpoint path's mtime.
        try:
            from handsneyes.platforms import load_adapter
            ckpt = load_adapter("linux_gnome").pointer_accel_checkpoint()
            if ckpt is not None:
                cfg = json.loads((ckpt / "config.json").read_text("utf-8"))
                args.label = cfg.get("platform", "current")
            else:
                args.label = "current"
        except Exception:
            args.label = "current"

    print(f"hitting {args.cc_url} with 8 canary clicks (label={args.label!r})")
    print(f"  timeout per click: {args.timeout:.0f}s")

    with httpx.Client(base_url=args.cc_url, timeout=args.timeout) as client:
        # Confirm cc is up.
        try:
            r = client.get("/api/state")
            r.raise_for_status()
        except Exception as e:
            print(f"ERROR: cc not reachable at {args.cc_url}: {e}", file=sys.stderr)
            return 2

        results = []
        for x, y in CANARY_POINTS:
            print(f"  → ({x:.2f}, {y:.2f}) …", flush=True)
            res = _hit_one(client, x, y)
            results.append(res)
            print(f"    steps={res['steps']} ({res['elapsed_s']:.1f}s)")

    summary = _print_summary(args.label, results)
    summary["points"] = [
        {**res, "x_pct": res["x"], "y_pct": res["y"]} for res in results
    ]

    if args.out:
        args.out.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        print(f"\nsummary written → {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
