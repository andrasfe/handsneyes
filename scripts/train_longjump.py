#!/usr/bin/env python3
"""train_longjump.py — fit a per-trajectory ``(target, cursor) → total HID``
MLP for the visual servo homer's first step.

The per-step ``pointer_accel`` model (v5) handles small residuals
well but only saw HID magnitudes ≤220 and per-step pixel deltas
≤19% in training. The homer's slam-to-target first step asks for
50%+ deltas with HIDs that sum to hundreds — far outside per-step
distribution.

This trains a SEPARATE model whose targets are the cumulative HID
across an ENTIRE successful trajectory. At runtime the homer can
query this once at the top of a click to get the full HID budget,
fire it as a chain of back-to-back bursts (≤127 per axis each, no
captures between), and only fall back to the closed-loop refinement
for the small residual that's left.

Inputs are 4-d:

    (target_dx_pct, target_dy_pct,
     initial_cursor_x_centred, initial_cursor_y_centred)

where ``target_dx_pct = target_x_pct - initial_cursor_x_pct`` and
``*_centred = pct * 2 - 1`` (shifted to [-1, 1]).

Outputs are 2-d total HID, normalised by ``HID_SCALE`` (default 500)
to roughly [-1, 1]. The runtime un-normalises and clamps to a
practical per-axis maximum (~1500 HID = 12 back-to-back max-bursts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HID_SCALE = 500.0


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _featurise(rows: list[dict], *, augment: bool):
    import numpy as np
    X = []
    Y = []
    for r in rows:
        cx = r.get("initial_cursor_x_pct")
        cy = r.get("initial_cursor_y_pct")
        tx = r.get("target_x_pct")
        ty = r.get("target_y_pct")
        if None in (cx, cy, tx, ty):
            continue
        dx = float(tx) - float(cx)
        dy = float(ty) - float(cy)
        hx = float(r["total_hid_dx"]) / HID_SCALE
        hy = float(r["total_hid_dy"]) / HID_SCALE
        cx_c = float(cx) * 2.0 - 1.0
        cy_c = float(cy) * 2.0 - 1.0
        samples = [(dx, dy, cx_c, cy_c, hx, hy)]
        if augment:
            # Sign-flip augmentation: the libinput curve is
            # x/y-symmetric, so reflecting input and output deltas
            # together is a valid training example. Quadruples the
            # effective dataset.
            samples.append((-dx, dy, -cx_c, cy_c, -hx, hy))
            samples.append((dx, -dy, cx_c, -cy_c, hx, -hy))
            samples.append((-dx, -dy, -cx_c, -cy_c, -hx, -hy))
        for d_x, d_y, c_x, c_y, h_x, h_y in samples:
            X.append([d_x, d_y, c_x, c_y])
            Y.append([h_x, h_y])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", type=Path, default=Path("data/ml/longjump"),
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/ml/checkpoints/longjump-v1"),
    )
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument(
        "--no-augment", dest="augment", action="store_false",
    )
    args = ap.parse_args()

    try:
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as e:
        print(f"missing deps: {e}", file=sys.stderr); return 2

    if not args.dataset.exists():
        print(f"dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2
    train_rows = _load_jsonl(args.dataset / "train.jsonl")
    val_rows = (
        _load_jsonl(args.dataset / "val.jsonl")
        if (args.dataset / "val.jsonl").exists() else []
    )
    if not train_rows:
        print("no train rows", file=sys.stderr); return 1
    Xtr, Ytr = _featurise(train_rows, augment=args.augment)
    Xv, Yv = (
        _featurise(val_rows, augment=False)
        if val_rows else (None, None)
    )
    print(
        f"augment={args.augment}  HID_SCALE={HID_SCALE}; "
        f"train shape: {Xtr.shape} → {Ytr.shape}; "
        f"val shape: {None if Xv is None else Xv.shape}"
    )

    class _MLP(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.fc1 = nn.Linear(4, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 2)

        def __call__(self, x):
            x = nn.gelu(self.fc1(x))
            x = nn.gelu(self.fc2(x))
            return self.fc3(x)

    model = _MLP(args.hidden)

    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr)

    n = Xtr.shape[0]
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(n)
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
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            tr = epoch_loss / max(1, n_batches)
            line = f"  epoch {epoch:>4d}/{args.epochs}  train_mse={tr:.6f}"
            if Xv is not None and Xv.shape[0] > 0:
                vl = float(loss_fn(model, mx.array(Xv), mx.array(Yv)))
                line += f"  val_mse={vl:.6f}"
            print(line)

    args.output.mkdir(parents=True, exist_ok=True)
    weights = {
        "fc1.weight": np.array(model.fc1.weight),
        "fc1.bias":   np.array(model.fc1.bias),
        "fc2.weight": np.array(model.fc2.weight),
        "fc2.bias":   np.array(model.fc2.bias),
        "fc3.weight": np.array(model.fc3.weight),
        "fc3.bias":   np.array(model.fc3.bias),
    }
    np.savez(str(args.output / "weights.npz"), **weights)
    (args.output / "config.json").write_text(json.dumps({
        "hidden": args.hidden,
        "direction": "longjump",
        "hid_scale": HID_SCALE,
        "input_features": [
            "target_dx_pct", "target_dy_pct",
            "initial_cursor_x_centred", "initial_cursor_y_centred",
        ],
        "output_features": ["total_hid_dx_norm", "total_hid_dy_norm"],
        "augmented": bool(args.augment),
        "train_rows": int(Xtr.shape[0]),
        "val_rows": int(0 if Xv is None else Xv.shape[0]),
        "platform": "ubuntu-libinput-adaptive",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved → {args.output}/weights.npz + config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
