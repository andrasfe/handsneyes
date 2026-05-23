#!/usr/bin/env python3
"""retrain_homer.py — re-train pointer_accel + longjump on accumulated
trajectories, gate the new checkpoints on a canary eval before
installing.

Pipeline:
  1. Build the per-step dataset (build_pointer_accel_dataset.py),
     excluding any trajectory_id in ``data/ml/canary/pointer_accel.jsonl``.
  2. Train pointer_accel into a NEW vN+1 directory.
  3. Score the new checkpoint on the canary. If median HID error
     exceeds the previous checkpoint's by more than
     ``--regress-tolerance`` (default 1.5×), REJECT and delete the
     new dir. Else accept.
  4. Same flow for longjump.
  5. Print a JSON summary so the cc retrain endpoint can stream
     the verdict back to the UI.

Exit code 0 if at least one model trained AND passed the canary
gate. Non-zero if everything regressed or training failed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def _next_vN(checkpoints_root: Path, prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}v(\d+)$")
    nmax = 0
    for d in checkpoints_root.glob(f"{prefix}v*"):
        m = pat.match(d.name)
        if m:
            nmax = max(nmax, int(m.group(1)))
    return nmax + 1


def _eval_pointer_accel(
    checkpoint_dir: Path, canary_path: Path,
) -> dict | None:
    if not canary_path.exists():
        return None
    try:
        from handsneyes.core.vision.pointer_accel import PointerAccelModel
    except Exception as e:
        print(f"  cannot import PointerAccelModel: {e}", file=sys.stderr)
        return None
    try:
        m = PointerAccelModel(checkpoint_dir)
    except Exception as e:
        print(f"  cannot load checkpoint: {e}", file=sys.stderr)
        return None
    errs_dx: list[float] = []
    errs_dy: list[float] = []
    with canary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cx = r.get("cursor_x_pct")
            cy = r.get("cursor_y_pct")
            if cx is None or cy is None:
                continue
            try:
                pred_dx, pred_dy = m.inverse(
                    target_dx_pct=r["measured_dx_pct"],
                    target_dy_pct=r["measured_dy_pct"],
                    cursor_x_pct=float(cx),
                    cursor_y_pct=float(cy),
                )
            except Exception:
                continue
            errs_dx.append(abs(pred_dx - r["hid_dx"]))
            errs_dy.append(abs(pred_dy - r["hid_dy"]))
    if not errs_dx:
        return None
    return {
        "n": len(errs_dx),
        "med_dx": statistics.median(errs_dx),
        "med_dy": statistics.median(errs_dy),
    }


def _eval_longjump(
    checkpoint_dir: Path, canary_path: Path,
) -> dict | None:
    if not canary_path.exists():
        return None
    try:
        from handsneyes.core.vision.longjump import LongJumpModel
    except Exception as e:
        print(f"  cannot import LongJumpModel: {e}", file=sys.stderr)
        return None
    try:
        m = LongJumpModel(checkpoint_dir)
    except Exception as e:
        print(f"  cannot load checkpoint: {e}", file=sys.stderr)
        return None
    errs_dx: list[float] = []
    errs_dy: list[float] = []
    with canary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                pred_dx, pred_dy = m.predict_total_hid(
                    cursor_x_pct=r["initial_cursor_x_pct"],
                    cursor_y_pct=r["initial_cursor_y_pct"],
                    target_x_pct=r["target_x_pct"],
                    target_y_pct=r["target_y_pct"],
                )
            except Exception:
                continue
            errs_dx.append(abs(pred_dx - r["total_hid_dx"]))
            errs_dy.append(abs(pred_dy - r["total_hid_dy"]))
    if not errs_dx:
        return None
    return {
        "n": len(errs_dx),
        "med_dx": statistics.median(errs_dx),
        "med_dy": statistics.median(errs_dy),
    }


def _run_step(cmd: list[str], cwd: Path) -> int:
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return proc.returncode


def _retrain_one(
    *,
    family: str,
    build_cmd: list[str],
    train_cmd_prefix: list[str],
    eval_fn,
    canary_path: Path,
    checkpoints_root: Path,
    cwd: Path,
    regress_tolerance: float,
) -> dict:
    result: dict = {"family": family}
    print(f"\n=== retrain {family}: build dataset ===")
    rc = _run_step(build_cmd, cwd)
    if rc != 0:
        result["status"] = "build_failed"
        return result
    n = _next_vN(checkpoints_root, f"{family}-")
    new_dir = checkpoints_root / f"{family}-v{n}"
    print(f"\n=== retrain {family}: train → {new_dir.name} ===")
    train_cmd = train_cmd_prefix + ["--output", str(new_dir)]
    rc = _run_step(train_cmd, cwd)
    if rc != 0:
        result["status"] = "train_failed"
        if new_dir.exists():
            shutil.rmtree(new_dir, ignore_errors=True)
        return result
    print(f"\n=== retrain {family}: canary eval ===")
    new_metrics = eval_fn(new_dir, canary_path)
    if new_metrics is None:
        result["status"] = "eval_failed"
        shutil.rmtree(new_dir, ignore_errors=True)
        return result
    result["new"] = {"checkpoint": new_dir.name, **new_metrics}
    prev = None
    for d in sorted(
        checkpoints_root.glob(f"{family}-v*"),
        key=lambda p: int(re.search(r"v(\d+)$", p.name).group(1)),
        reverse=True,
    ):
        if d != new_dir and (d / "config.json").exists():
            prev = d
            break
    if prev is not None:
        prev_metrics = eval_fn(prev, canary_path)
        result["previous"] = {
            "checkpoint": prev.name,
            **(prev_metrics or {}),
        }
        if prev_metrics is not None:
            new_score = new_metrics["med_dx"] + new_metrics["med_dy"]
            prev_score = prev_metrics["med_dx"] + prev_metrics["med_dy"]
            if new_score > prev_score * regress_tolerance:
                print(
                    f"  REJECTED: new score {new_score:.1f} > "
                    f"{regress_tolerance}× previous {prev_score:.1f}"
                )
                result["status"] = "rejected_regression"
                result["new_score"] = new_score
                result["prev_score"] = prev_score
                shutil.rmtree(new_dir, ignore_errors=True)
                return result
            print(
                f"  ACCEPTED: new score {new_score:.1f} vs "
                f"previous {prev_score:.1f}"
            )
            result["new_score"] = new_score
            result["prev_score"] = prev_score
    else:
        print("  no previous checkpoint to compare; accepting unconditionally")
    result["status"] = "accepted"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoints-root", type=Path,
        default=Path("data/ml/checkpoints"),
    )
    ap.add_argument(
        "--canary-dir", type=Path, default=Path("data/ml/canary"),
    )
    ap.add_argument(
        "--regress-tolerance", type=float, default=1.5,
        help="Reject the new checkpoint if its canary score is more "
             "than this multiple of the previous checkpoint's score. "
             "Default 1.5 — a 50%% regression triggers rejection.",
    )
    ap.add_argument(
        "--summary-out", type=Path, default=None,
        help="Write a JSON summary of the verdict here. Used by the "
             "cc retrain endpoint to stream the result back to the UI.",
    )
    ap.add_argument(
        "--only", choices=["pointer_accel", "longjump", "both"],
        default="both",
    )
    ap.add_argument(
        "--hsv-only", action="store_true",
        help="Pass --hsv-only to build_pointer_accel_dataset.py. Only "
             "useful with a redglass-style cursor; on Yaru white this "
             "would empty the corpus.",
    )
    args = ap.parse_args()

    cwd = Path.cwd()
    py = sys.executable

    summary: dict = {"results": []}

    if args.only in ("pointer_accel", "both"):
        build_cmd = [py, "scripts/build_pointer_accel_dataset.py"]
        if args.hsv_only:
            build_cmd.append("--hsv-only")
        verdict = _retrain_one(
            family="pointer_accel",
            build_cmd=build_cmd,
            train_cmd_prefix=[py, "scripts/train_pointer_accel.py"],
            eval_fn=_eval_pointer_accel,
            canary_path=args.canary_dir / "pointer_accel.jsonl",
            checkpoints_root=args.checkpoints_root,
            cwd=cwd,
            regress_tolerance=args.regress_tolerance,
        )
        summary["results"].append(verdict)

    if args.only in ("longjump", "both"):
        verdict = _retrain_one(
            family="longjump",
            build_cmd=[py, "scripts/build_longjump_dataset.py"],
            train_cmd_prefix=[py, "scripts/train_longjump.py"],
            eval_fn=_eval_longjump,
            canary_path=args.canary_dir / "longjump.jsonl",
            checkpoints_root=args.checkpoints_root,
            cwd=cwd,
            regress_tolerance=args.regress_tolerance,
        )
        summary["results"].append(verdict)

    summary["any_accepted"] = any(
        r.get("status") == "accepted" for r in summary["results"]
    )

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
    return 0 if summary["any_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
