#!/usr/bin/env python3
"""build_longjump_dataset.py — collect per-trajectory ``(initial cursor,
target) → cumulative HID`` rows for training a long-jump model.

The per-step ``pointer_accel`` model handles small residual moves
inside the visual servo loop, but the FIRST step of a homing run
asks it to predict an HID for a slam-to-target move that's typically
40-60% of image width. v5's training data only went up to ~19% per
step — that first step is solid extrapolation, which is why even a
v5-equipped homer still needs 7-10 closed-loop iterations to converge.

This dataset reframes the problem at the trajectory level:

  Input:  initial cursor position + target position
  Output: TOTAL HID delta (cum-sum across all successful steps) that
          actually got the cursor to the target

A model trained on this can be queried once at the top of a click to
get the full HID budget; the runtime can then fire that budget as a
chain of back-to-back bursts (each ≤127 per axis) without per-step
captures, and only fall back to closed-loop for the fine residual.

Output schema (per row)::

    {
      "trajectory_id": "<run>/homer/<vs-id>",
      "n_steps": 8,
      "initial_cursor_x_pct": 0.04,
      "initial_cursor_y_pct": 0.04,
      "target_x_pct": 0.52,
      "target_y_pct": 0.43,
      "total_hid_dx": 312,
      "total_hid_dy": 196,
      "final_residual_pct": 0.004,
    }

Splits 80/10/10 by trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


def _iter_history_files(runs_root: Path):
    return runs_root.glob("**/homer/*/history.jsonl")


def _trajectory_row(traj_id: str, steps: list[dict]) -> dict | None:
    """Aggregate one trajectory's step rows into a single long-jump
    training row, or return None if the trajectory is unusable.

    Unusable cases:
      - No steps.
      - Initial cursor position missing (oscillation detection failed).
      - Trajectory doesn't end with a click (didn't converge).
      - Target inconsistent across steps (shouldn't happen, but
        defensive — bad data, drop).
    """
    if not steps:
        return None
    first = steps[0]
    last = steps[-1]
    init_cursor = first.get("cursor_img")
    if not (isinstance(init_cursor, list) and len(init_cursor) == 2):
        return None
    target = last.get("target_img")
    if not (isinstance(target, list) and len(target) == 2):
        return None
    # Only successful trajectories — note containing "click_sent"
    # means the homer's geometric gate fired.
    note = (last.get("note") or "").lower()
    if "click_sent" not in note:
        return None
    # Sum HID across the trajectory. Skip the click row itself
    # (hid=0, just the click event).
    total_hid_dx = 0
    total_hid_dy = 0
    n = 0
    for s in steps:
        hdx = s.get("hid_dx") or 0
        hdy = s.get("hid_dy") or 0
        if hdx == 0 and hdy == 0:
            continue
        total_hid_dx += int(hdx)
        total_hid_dy += int(hdy)
        n += 1
    if n == 0:
        return None
    # final_residual_pct: how close did the last detected cursor
    # land to target — a quality signal for sanity-check filters.
    final_cursor = last.get("cursor_img")
    if isinstance(final_cursor, list) and len(final_cursor) == 2:
        final_residual = math.hypot(
            target[0] - final_cursor[0],
            target[1] - final_cursor[1],
        )
    else:
        final_residual = None
    return {
        "trajectory_id": traj_id,
        "n_steps": n,
        "initial_cursor_x_pct": float(init_cursor[0]),
        "initial_cursor_y_pct": float(init_cursor[1]),
        "target_x_pct": float(target[0]),
        "target_y_pct": float(target[1]),
        "total_hid_dx": total_hid_dx,
        "total_hid_dy": total_hid_dy,
        "final_residual_pct": (
            None if final_residual is None else float(final_residual)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/handsneyes/runs",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("data/ml/longjump"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--since", type=str, default=None,
        help="Drop history.jsonl files with mtime < this epoch "
             "(or ISO timestamp).",
    )
    ap.add_argument(
        "--max-final-residual", type=float, default=0.02,
        help="Discard trajectories whose final cursor landed > this "
             "fraction of the image from the target. Default 0.02 (2%%, "
             "~38 px on 1920×1080) — clean clicks only.",
    )
    ap.add_argument(
        "--exclude-canary", type=Path,
        default=Path("data/ml/canary/longjump.jsonl"),
        help="Exclude any trajectory_id present in this canary "
             "file from the training corpus. Prevents training "
             "data / eval-set overlap. Set empty to skip.",
    )
    args = ap.parse_args()
    since_ts: float | None = None
    if args.since:
        try:
            since_ts = float(args.since)
        except ValueError:
            from datetime import datetime
            since_ts = datetime.fromisoformat(args.since).timestamp()

    canary_traj_ids: set[str] = set()
    if args.exclude_canary and args.exclude_canary.exists():
        with args.exclude_canary.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    canary_traj_ids.add(json.loads(line)["trajectory_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    if not args.runs_root.exists():
        print(f"runs root not found: {args.runs_root}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    n_files = 0
    n_lines = 0
    n_dropped_canary = 0
    n_dropped_age = 0
    n_dropped_quality = 0
    for hist_path in _iter_history_files(args.runs_root):
        if since_ts is not None:
            try:
                if hist_path.stat().st_mtime < since_ts:
                    n_dropped_age += 1
                    continue
            except OSError:
                continue
        n_files += 1
        traj_id = str(hist_path.parent.relative_to(args.runs_root))
        if traj_id in canary_traj_ids:
            n_dropped_canary += 1
            continue
        steps: list[dict] = []
        try:
            with hist_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
        row = _trajectory_row(traj_id, steps)
        if row is None:
            continue
        # Quality gate on final residual.
        if (row["final_residual_pct"] is not None
                and row["final_residual_pct"] > args.max_final_residual):
            n_dropped_quality += 1
            continue
        rows.append(row)

    print(
        f"scanned {n_files} history.jsonl file(s), {n_lines} step "
        f"line(s) → {len(rows)} usable trajectory row(s)"
    )
    if n_dropped_age:
        print(f"  dropped {n_dropped_age} pre-{args.since} file(s)")
    if n_dropped_quality:
        print(
            f"  dropped {n_dropped_quality} trajectory(s) with final "
            f"residual > {args.max_final_residual:.0%}"
        )
    if n_dropped_canary:
        print(f"  dropped {n_dropped_canary} canary trajectory(s)")
    if not rows:
        print(
            "no usable rows — run scripts/collect_pointer_accel.sh first.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_val = max(1, n // 10) if n >= 10 else 0
    n_test = max(1, n // 10) if n >= 10 else 0
    val_rows = rows[:n_val]
    test_rows = rows[n_val: n_val + n_test]
    train_rows = rows[n_val + n_test:]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in (
        ("train", train_rows), ("val", val_rows), ("test", test_rows),
    ):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "total_rows": len(rows),
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows → {args.out}")
    for name, items in (
        ("train", train_rows), ("val", val_rows), ("test", test_rows),
    ):
        print(f"  {name:>5}: {len(items):>5d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
