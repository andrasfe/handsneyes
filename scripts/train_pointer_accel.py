#!/usr/bin/env python3
"""train_pointer_accel.py — fit a tiny MLX MLP that maps HID deltas
to observed cursor deltas under Ubuntu's pointer-acceleration curve.

Forward model::

    f((hid_dx, hid_dy, cursor_x_pct, cursor_y_pct))
        → (measured_dx_pct, measured_dy_pct)

The inverse — "given a target pixel delta, what HID delta should I
send" — is the actually useful thing for the homer. We get it by
running Newton-style root-finding on the trained forward model
(see :class:`handsneyes.commander.pointer_accel.PointerAccelModel`).

Why MLP and not a fitted closed-form curve: Ubuntu's libinput
"adaptive" profile is piecewise non-linear AND velocity-dependent
in subtle ways (acceleration scales with sqrt(dx²+dy²)). A 2-layer
MLP with ~50 hidden units fits it cleanly from a few hundred
samples; a closed-form fit would need careful per-axis parameter
search.

Inputs are 4-d: (hid_dx, hid_dy, cursor_x_pct, cursor_y_pct), all
normalised to roughly ``[-1, 1]``. HID values come in ``[-127, 127]``
so we divide by 127. Cursor positions are already in ``[0, 1]`` and
we shift to ``[-1, 1]``.

Outputs are 2-d: measured_dx_pct, measured_dy_pct (already in pct).

Usage::

    python scripts/train_pointer_accel.py \\
        --dataset data/ml/pointer_accel \\
        --output  data/ml/checkpoints/pointer_accel-v1 \\
        --epochs  400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _featurise(rows: list[dict], *, inverse: bool, augment: bool):
    """Build (X, Y) for either the forward or inverse problem.

    Forward: ``(hid, cursor) → measured`` — what we used to do; the
    homer then inverts at runtime via Newton's method (finite-diff
    Jacobian, 6 forward passes, noisy near zero).

    Inverse (default in v3): ``(measured, cursor) → hid``. We have the
    inverse samples for free — every row of the dataset is a real
    ``(target HID, observed delta)`` pair, so flipping the I/O turns
    the same data into an inverse-training corpus. Inference becomes
    a single forward pass; no Newton iteration, no Jacobian noise.

    Augment: the pointer-accel curve is symmetric under (dx → -dx)
    and (dy → -dy) independently. Adding the three sign-flipped
    copies roughly 4× the dataset for free, and forces the model to
    learn that symmetry instead of memorising the specific positive-
    quadrant trajectories the homer happens to produce.
    """
    import numpy as np
    X = []
    Y = []
    for r in rows:
        cx = r.get("cursor_x_pct")
        cy = r.get("cursor_y_pct")
        if cx is None or cy is None:
            continue
        hx = r["hid_dx"] / 127.0
        hy = r["hid_dy"] / 127.0
        mx = float(r["measured_dx_pct"])
        my = float(r["measured_dy_pct"])
        cx_c = (cx * 2.0) - 1.0
        cy_c = (cy * 2.0) - 1.0
        samples = [(hx, hy, mx, my, cx_c, cy_c)]
        if augment:
            samples.append((-hx, hy, -mx, my, -cx_c, cy_c))
            samples.append((hx, -hy, mx, -my, cx_c, -cy_c))
            samples.append((-hx, -hy, -mx, -my, -cx_c, -cy_c))
        for hx_, hy_, mx_, my_, cx_, cy_ in samples:
            if inverse:
                # Inverse direction: tell me what HID to send.
                X.append([mx_, my_, cx_, cy_])
                Y.append([hx_, hy_])
            else:
                X.append([hx_, hy_, cx_, cy_])
                Y.append([mx_, my_])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", type=Path,
        default=Path("data/ml/pointer_accel"),
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/ml/checkpoints/pointer_accel-v1"),
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--inverse", action="store_true", default=True,
        help="Train (measured, cursor)→hid directly (default).",
    )
    ap.add_argument(
        "--forward", dest="inverse", action="store_false",
        help="Train forward (hid, cursor)→measured; runtime inverts.",
    )
    ap.add_argument(
        "--augment", action="store_true", default=True,
        help="Sign-flip augmentation (x4 dataset; default on).",
    )
    ap.add_argument(
        "--no-augment", dest="augment", action="store_false",
    )
    ap.add_argument(
        "--init-from", type=Path, default=None,
        help="Warm-start from an existing checkpoint dir. Loads its "
             "weights.npz as the initial state, then continues training "
             "on the current dataset. Use --lr ~5e-4 (10x lower than "
             "from-scratch) when warm-starting. Useful for fine-tuning "
             "a known-good model on small batches of new data instead "
             "of retraining from random init.",
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
    Xtr, Ytr = _featurise(
        train_rows, inverse=args.inverse, augment=args.augment,
    )
    if val_rows:
        # Don't augment val — we measure on real-world (non-flipped)
        # rows only so the reported MSE reflects deployment.
        Xv, Yv = _featurise(val_rows, inverse=args.inverse, augment=False)
    else:
        Xv, Yv = None, None
    mode = "inverse (measured,cursor)→hid" if args.inverse \
        else "forward (hid,cursor)→measured"
    print(
        f"mode: {mode}; augment={args.augment}; "
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

    if args.init_from is not None:
        init_npz = args.init_from / "weights.npz"
        init_cfg = args.init_from / "config.json"
        if not init_npz.exists():
            print(f"--init-from: weights.npz missing at {init_npz}",
                  file=sys.stderr)
            return 2
        # Verify the hidden size matches; warm-starting a different
        # shape would silently broadcast or fail mid-train.
        try:
            cfg = json.loads(init_cfg.read_text("utf-8"))
            init_hidden = int(cfg.get("hidden", -1))
            if init_hidden != args.hidden:
                print(
                    f"--init-from hidden={init_hidden} != --hidden "
                    f"{args.hidden}; refusing to load mismatched shape.",
                    file=sys.stderr,
                )
                return 2
        except Exception:
            pass  # no config — best-effort load
        w = np.load(init_npz)
        model.fc1.weight = mx.array(w["fc1.weight"].astype(np.float32))
        model.fc1.bias = mx.array(w["fc1.bias"].astype(np.float32))
        model.fc2.weight = mx.array(w["fc2.weight"].astype(np.float32))
        model.fc2.bias = mx.array(w["fc2.bias"].astype(np.float32))
        model.fc3.weight = mx.array(w["fc3.weight"].astype(np.float32))
        model.fc3.bias = mx.array(w["fc3.bias"].astype(np.float32))
        mx.eval(model.parameters())
        print(f"warm-started from {args.init_from}")

    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr)

    n = Xtr.shape[0]
    rng = np.random.default_rng(0)
    # Track the best-val checkpoint so we save the early-stopping
    # weights, not the last-epoch weights (which often overfit when
    # the dataset is small).
    best_val = float("inf")
    best_weights = None
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
        # Track val + snapshot best weights every epoch (cheap).
        if Xv is not None and Xv.shape[0] > 0:
            vl = float(loss_fn(model, mx.array(Xv), mx.array(Yv)))
            if vl < best_val:
                best_val = vl
                best_weights = {
                    "fc1.weight": np.array(model.fc1.weight),
                    "fc1.bias":   np.array(model.fc1.bias),
                    "fc2.weight": np.array(model.fc2.weight),
                    "fc2.bias":   np.array(model.fc2.bias),
                    "fc3.weight": np.array(model.fc3.weight),
                    "fc3.bias":   np.array(model.fc3.bias),
                    "_epoch": epoch,
                }
        else:
            vl = None
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            tr = epoch_loss / max(1, n_batches)
            line = f"  epoch {epoch:>4d}/{args.epochs}  train_mse={tr:.6f}"
            if vl is not None:
                line += f"  val_mse={vl:.6f}"
                line += f"  best_val={best_val:.6f}@{best_weights['_epoch'] if best_weights else '-'}"
            print(line)

    # Persist weights as a tiny safetensors / numpy bundle. We use a
    # simple dict so the runtime wrapper doesn't need mlx_lm's
    # heavier checkpoint utilities.
    args.output.mkdir(parents=True, exist_ok=True)
    # Prefer the best-val checkpoint when we have validation data.
    if best_weights is not None:
        print(
            f"saving best-val checkpoint from epoch "
            f"{best_weights['_epoch']} (val_mse={best_val:.6f}) "
            f"instead of last-epoch weights"
        )
        weights = {k: v for k, v in best_weights.items() if k != "_epoch"}
    else:
        weights = {
            "fc1.weight": np.array(model.fc1.weight),
            "fc1.bias":   np.array(model.fc1.bias),
            "fc2.weight": np.array(model.fc2.weight),
            "fc2.bias":   np.array(model.fc2.bias),
            "fc3.weight": np.array(model.fc3.weight),
            "fc3.bias":   np.array(model.fc3.bias),
        }
    np.savez(str(args.output / "weights.npz"), **weights)
    if args.inverse:
        input_features = [
            "measured_dx_pct", "measured_dy_pct",
            "cursor_x_centred", "cursor_y_centred",
        ]
        output_features = ["hid_dx_norm", "hid_dy_norm"]
    else:
        input_features = [
            "hid_dx_norm", "hid_dy_norm",
            "cursor_x_centred", "cursor_y_centred",
        ]
        output_features = ["measured_dx_pct", "measured_dy_pct"]
    (args.output / "config.json").write_text(json.dumps({
        "hidden": args.hidden,
        "direction": "inverse" if args.inverse else "forward",
        "input_features": input_features,
        "output_features": output_features,
        "augmented": bool(args.augment),
        "train_rows": int(Xtr.shape[0]),
        "val_rows": int(0 if Xv is None else Xv.shape[0]),
        "platform": "ubuntu-libinput-adaptive",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved → {args.output}/weights.npz + config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
