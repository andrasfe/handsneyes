#!/usr/bin/env python3
"""build_pointer_accel_dataset.py — collect homer step records into
a flat training table for the open-loop forward model.

Walks every ``<run>/homer/<*>/history.jsonl`` produced by
``VisualServoHomer`` (persisted via ``_record_step``) and emits one
``data/ml/pointer_accel/{train,val,test}.jsonl`` row per usable step.

Output schema (per row)::

    {
      "trajectory_id": "<run>/homer/<vs-id>",
      "step_idx": 4,
      "hid_dx": 18,            // sent HID delta x (signed)
      "hid_dy": -7,             // sent HID delta y (signed)
      "measured_dx_pct": 0.041, // observed cursor delta x (normalised)
      "measured_dy_pct": -0.018,// observed cursor delta y (normalised)
      "cursor_x_pct": 0.523,    // cursor position BEFORE the step
      "cursor_y_pct": 0.418,
      "note": "hsv_measured"
    }

The "before" cursor position lets a future model condition on edge
proximity (pointer accel is identity-shaped in the middle and gets
weird near borders).

Filtering:
  * Drop steps where ``hid_dx == 0 && hid_dy == 0`` (those are post-
    click "confirm" records, not pointer-accel samples).
  * Drop steps without a measured delta. The homer logs a measured
    delta even when HSV detection fails (note=openloop_fallback)
    via frame-diff fallback — those rows are NOISIER but still
    carry the right shape of the curve, so we keep them and let
    the training MSE handle the noise.

Splits 80/10/10 by trajectory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _iter_history_files(runs_root: Path):
    return runs_root.glob("**/homer/*/history.jsonl")


def _row_from_step(
    traj_id: str,
    idx: int,
    step: dict,
    *,
    ratio_min: float = 3e-4,
    ratio_max: float = 8e-3,
) -> dict | None:
    hid_dx = step.get("hid_dx")
    hid_dy = step.get("hid_dy")
    if hid_dx is None or hid_dy is None:
        return None
    if hid_dx == 0 and hid_dy == 0:
        return None
    mdx = step.get("measured_dx_pct")
    mdy = step.get("measured_dy_pct")
    if mdx is None or mdy is None:
        return None
    # Sanity gate on the per-axis pct-per-hid ratio.
    #
    # Default [3e-4, 8e-3] matches the Linux libinput-adaptive curve.
    # macOS uses a different acceleration profile — pass
    # `--ratio-min 1e-5 --ratio-max 2e-2` for that. Rows outside the
    # gate are almost certainly corrupted (operator moved the mouse
    # mid-click, cursor hit a screen edge, network glitch dropped
    # the HID, etc.) and would poison the next training round.
    for h, m in ((hid_dx, mdx), (hid_dy, mdy)):
        if abs(h) < 3:
            # Skip axes with tiny HIDs — they're dominated by noise
            # in the measured delta and the ratio is meaningless.
            continue
        ratio = abs(float(m) / float(h))
        if ratio < ratio_min or ratio > ratio_max:
            return None
    cursor = step.get("cursor_img")
    cx = cy = None
    if isinstance(cursor, list) and len(cursor) == 2:
        cx, cy = float(cursor[0]), float(cursor[1])
    return {
        "trajectory_id": traj_id,
        "step_idx": idx,
        "hid_dx": int(hid_dx),
        "hid_dy": int(hid_dy),
        "measured_dx_pct": float(mdx),
        "measured_dy_pct": float(mdy),
        "cursor_x_pct": cx,
        "cursor_y_pct": cy,
        "note": step.get("note", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/handsneyes/runs",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("data/ml/pointer_accel"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hsv-only", action="store_true",
        help="Keep only rows where HSV cursor detection succeeded "
             "(note=hsv_measured). Frame-diff fallback rows are "
             "noisier; with the redglass cursor on the target we "
             "get HSV rows directly.",
    )
    ap.add_argument(
        "--since", type=str, default=None,
        help="Drop history.jsonl files older than this mtime "
             "(ISO timestamp or epoch). Use after switching cursor "
             "theme to filter out pre-theme runs.",
    )
    ap.add_argument(
        "--exclude-canary", type=Path,
        default=Path("data/ml/canary/pointer_accel.jsonl"),
        help="Exclude any trajectory_id present in this canary "
             "file from the training corpus. Prevents training data "
             "from overlapping the eval set used to gate retrains. "
             "Set empty to skip.",
    )
    ap.add_argument(
        "--exclude-pattern", action="append", default=[],
        help="Glob fragment to skip when walking runs. Match is done "
             "against each history.jsonl's parent dir name + the "
             "session dir name (e.g. 'explore_v2*'). Repeatable.",
    )
    ap.add_argument(
        "--platform", type=str, default=None,
        help="Keep only rows tagged with this platform (e.g. "
             "'linux_gnome', 'macos'). Untagged rows (written before "
             "platform tagging existed) are treated as 'linux_gnome' "
             "since every legacy row is from the Yaru target. Set "
             "empty / unset to include all platforms (legacy behaviour, "
             "DANGEROUS when training a non-linux model since IOHID "
             "and libinput curves don't mix).",
    )
    ap.add_argument(
        "--ratio-min", type=float, default=3e-4,
        help="Lower bound on |measured_pct / hid|. Default 3e-4 "
             "matches Linux libinput-adaptive. macOS IOHID needs "
             "a wider gate — try 1e-5.",
    )
    ap.add_argument(
        "--ratio-max", type=float, default=8e-3,
        help="Upper bound on |measured_pct / hid|. Default 8e-3 "
             "matches Linux libinput-adaptive. macOS needs ~2e-2.",
    )
    args = ap.parse_args()
    since_ts = None
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

    import fnmatch

    rows: list[dict] = []
    n_files = 0
    n_lines = 0
    n_dropped_note = 0
    n_dropped_age = 0
    n_dropped_canary = 0
    n_dropped_pattern = 0
    n_dropped_platform = 0
    for hist_path in _iter_history_files(args.runs_root):
        if since_ts is not None:
            try:
                if hist_path.stat().st_mtime < since_ts:
                    n_dropped_age += 1
                    continue
            except OSError:
                continue
        # Skip excluded session dirs (e.g. explore_v2*).
        if args.exclude_pattern:
            rel = hist_path.relative_to(args.runs_root)
            parts = rel.parts
            skip = any(
                fnmatch.fnmatch(part, pat)
                for part in parts
                for pat in args.exclude_pattern
            )
            if skip:
                n_dropped_pattern += 1
                continue
        n_files += 1
        traj_id = str(hist_path.parent.relative_to(args.runs_root))
        if traj_id in canary_traj_ids:
            n_dropped_canary += 1
            continue
        try:
            with hist_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        step = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    r = _row_from_step(
                        traj_id, i, step,
                        ratio_min=args.ratio_min,
                        ratio_max=args.ratio_max,
                    )
                    if r is None:
                        continue
                    if args.hsv_only and r.get("note") != "hsv_measured":
                        n_dropped_note += 1
                        continue
                    if args.platform:
                        # Untagged rows are legacy linux_gnome.
                        row_platform = step.get("platform", "linux_gnome")
                        if row_platform != args.platform:
                            n_dropped_platform += 1
                            continue
                        r["platform"] = row_platform
                    rows.append(r)
        except OSError:
            continue
    if since_ts is not None:
        print(f"dropped {n_dropped_age} pre-{args.since} file(s)")
    if args.hsv_only:
        print(f"dropped {n_dropped_note} non-HSV row(s)")
    if n_dropped_canary:
        print(f"dropped {n_dropped_canary} canary trajectory(s)")
    if n_dropped_pattern:
        print(f"dropped {n_dropped_pattern} excluded-pattern file(s)")
    if args.platform:
        print(f"kept rows for platform={args.platform!r}; "
              f"dropped {n_dropped_platform} row(s) from other platforms")

    print(
        f"scanned {n_files} history.jsonl file(s), {n_lines} step "
        f"line(s) → {len(rows)} usable training row(s)"
    )
    if not rows:
        print(
            "no usable rows — run scripts/collect_pointer_accel.sh first.",
            file=sys.stderr,
        )
        return 1

    by_traj: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_traj[r["trajectory_id"]].append(r)
    traj_ids = sorted(by_traj.keys())
    rng = random.Random(args.seed)
    rng.shuffle(traj_ids)
    n = len(traj_ids)
    n_val = max(1, n // 10) if n >= 10 else 0
    n_test = max(1, n // 10) if n >= 10 else 0
    val_ids = set(traj_ids[:n_val])
    test_ids = set(traj_ids[n_val: n_val + n_test])
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for tid, items in by_traj.items():
        if tid in val_ids:
            splits["val"].extend(items)
        elif tid in test_ids:
            splits["test"].extend(items)
        else:
            splits["train"].extend(items)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (args.out / f"{name}.jsonl").open(
            "w", encoding="utf-8",
        ) as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "total_rows": len(rows),
        "total_trajectories": len(by_traj),
        "splits": {
            name: {
                "rows": len(items),
                "trajectories": len({
                    r["trajectory_id"] for r in items
                }),
            } for name, items in splits.items()
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} rows across {len(by_traj)} trajectories "
        f"→ {args.out}"
    )
    for name, items in splits.items():
        print(f"  {name:>5}: {len(items):>5d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
