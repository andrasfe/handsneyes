#!/usr/bin/env python3
"""train_cursor_detector.py — single-frame cursor detector (MLX).

Input: one webcam frame, grayscale, downsampled to 384x216.
Output: (x_pct, y_pct) ∈ [0, 1]^2.

Architecture (small CNN + spatial-softmax head, DSNT-style):
  - 4 conv layers (strides 2, 2, 1, 1) → 96x54 feature map
  - 1x1 conv → 1-channel heatmap
  - spatial softmax + expected-coordinate readout → (x, y)

The DSNT readout gives sub-pixel accuracy from a coarse heatmap and
is end-to-end differentiable. Loss is plain MSE on coordinates.

Training data: see scripts/collect_cursor_detector_labels.py.

Output: data/ml/checkpoints/cursor_detector-vN/{weights.npz, config.json}
The shipped runtime loader is core/vision/cursor_detector.py
(numpy-only inference, no MLX on the hot path).

Usage::

    python scripts/train_cursor_detector.py \\
        --dataset data/ml/cursor_detector/<session_ts> \\
        --epochs 80 \\
        --output data/ml/checkpoints/cursor_detector-v1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Input dims kept fixed for the runtime — change here, change the
# loader, change the homer integration.
_IN_H = 216
_IN_W = 384


def _load_labels(session_dir: Path) -> list[dict]:
    labels_path = session_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_frame(session_dir: Path, frame_path: str) -> np.ndarray:
    """Load a frame, grayscale + resize to (IN_H, IN_W). Returns
    a float32 array in [0, 1] of shape (IN_H, IN_W, 1)."""
    full = session_dir / frame_path
    img = cv2.imread(str(full), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not read {full}")
    img = cv2.resize(img, (_IN_W, _IN_H), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)[..., None]  # NHWC


def _featurise(
    session_dir: Path, rows: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(rows), _IN_H, _IN_W, 1), dtype=np.float32)
    Y = np.empty((len(rows), 2), dtype=np.float32)
    for i, r in enumerate(rows):
        X[i] = _load_frame(session_dir, r["frame"])
        Y[i, 0] = float(r["x_pct"])
        Y[i, 1] = float(r["y_pct"])
    return X, Y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Session dir from collect_cursor_detector_labels.py.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Checkpoint output dir.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as e:
        print(f"missing deps (mlx): {e}", file=sys.stderr); return 2

    if not args.dataset.exists():
        print(f"dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2
    rows = _load_labels(args.dataset)
    if len(rows) < 20:
        print(f"only {len(rows)} labels — collect more first", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"loaded {len(rows)} labels  train={len(train_rows)} val={len(val_rows)}")
    print("loading frames…", flush=True)
    Xtr, Ytr = _featurise(args.dataset, train_rows)
    Xv, Yv = _featurise(args.dataset, val_rows)
    print(f"  train: {Xtr.shape}  val: {Xv.shape}")

    class _CursorDet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
            self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.c3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
            self.c4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.head = nn.Conv2d(64, 1, 1)

        def __call__(self, x):
            x = nn.gelu(self.c1(x))
            x = nn.gelu(self.c2(x))
            x = nn.gelu(self.c3(x))
            x = nn.gelu(self.c4(x))
            return self.head(x)  # (N, h, w, 1)

    # Pre-computed coord grids for the DSNT readout. Heatmap is
    # 96x54 (input 384x216 with two stride-2 layers).
    H_OUT = _IN_H // 4
    W_OUT = _IN_W // 4
    gx = mx.array(np.linspace(0, 1, W_OUT, dtype=np.float32))  # (W,)
    gy = mx.array(np.linspace(0, 1, H_OUT, dtype=np.float32))  # (H,)

    def readout(heatmap):
        """heatmap: (N, H, W, 1) → (N, 2)."""
        N = heatmap.shape[0]
        h = heatmap.reshape((N, H_OUT * W_OUT))
        p = mx.softmax(h, axis=-1).reshape((N, H_OUT, W_OUT))
        # Marginals along each axis.
        px = mx.sum(p, axis=1)  # (N, W)
        py = mx.sum(p, axis=2)  # (N, H)
        x = mx.sum(px * gx, axis=-1)
        y = mx.sum(py * gy, axis=-1)
        return mx.stack([x, y], axis=-1)

    model = _CursorDet()

    def loss_fn(model, x, y):
        heatmap = model(x)
        pred = readout(heatmap)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr)

    n = Xtr.shape[0]
    np_rng = np.random.default_rng(args.seed)
    best_val = float("inf")
    best_weights: dict | None = None

    for epoch in range(1, args.epochs + 1):
        idx = np_rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            j = idx[start: start + args.batch_size]
            xb = mx.array(Xtr[j])
            yb = mx.array(Ytr[j])
            loss, grads = loss_and_grad(model, xb, yb)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            epoch_loss += float(loss)
            n_batches += 1
        # Validation (small set, just do full pass)
        vl_total = 0.0
        vb = 0
        for vstart in range(0, Xv.shape[0], args.batch_size):
            xb = mx.array(Xv[vstart: vstart + args.batch_size])
            yb = mx.array(Yv[vstart: vstart + args.batch_size])
            vl_total += float(loss_fn(model, xb, yb))
            vb += 1
        val_loss = vl_total / max(1, vb)
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {
                "c1.weight": np.array(model.c1.weight),
                "c1.bias":   np.array(model.c1.bias),
                "c2.weight": np.array(model.c2.weight),
                "c2.bias":   np.array(model.c2.bias),
                "c3.weight": np.array(model.c3.weight),
                "c3.bias":   np.array(model.c3.bias),
                "c4.weight": np.array(model.c4.weight),
                "c4.bias":   np.array(model.c4.bias),
                "head.weight": np.array(model.head.weight),
                "head.bias":   np.array(model.head.bias),
                "_epoch": epoch,
                "_val_mse": val_loss,
            }
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            tr = epoch_loss / max(1, n_batches)
            # Convert MSE on normalized coords to a rough pixel-error
            # estimate at 1920x1080 (geometric mean of axes).
            px_err = float(np.sqrt(val_loss)) * np.sqrt(1920 * 1080)
            print(
                f"  epoch {epoch:>3d}/{args.epochs}  train_mse={tr:.6f} "
                f"val_mse={val_loss:.6f} (~{px_err:.1f}px) "
                f"best={best_val:.6f}@{best_weights['_epoch']}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    assert best_weights is not None
    # Strip the metadata keys before saving as a clean .npz.
    epoch_at_best = best_weights.pop("_epoch")
    val_at_best = best_weights.pop("_val_mse")
    np.savez(args.output / "weights.npz", **best_weights)
    (args.output / "config.json").write_text(json.dumps({
        "platform": "cursor-detector-dsnt-v1",
        "in_h": _IN_H, "in_w": _IN_W,
        "head_h": H_OUT, "head_w": W_OUT,
        "best_epoch": epoch_at_best,
        "best_val_mse": float(val_at_best),
        "train_rows": int(Xtr.shape[0]),
        "val_rows": int(Xv.shape[0]),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved best-val checkpoint (epoch {epoch_at_best}, "
          f"val_mse={val_at_best:.6f}) → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
