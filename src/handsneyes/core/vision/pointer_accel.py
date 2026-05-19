# ruff: noqa: N806
#   J = Jacobian, conventional uppercase in Newton-iteration / linalg code.
"""Open-loop pointer-acceleration model.

A tiny 2-layer MLP trained by ``scripts/train_pointer_accel.py``
that maps ``(hid_dx, hid_dy, cursor_x_pct, cursor_y_pct)`` to the
*observed* cursor delta in normalised image coordinates (the
quantity Ubuntu's libinput "adaptive" acceleration profile
multiplies into the raw HID delta).

Used by the visual-servo homer to do a one-shot open-loop move
instead of the closed-loop ratio-learning iteration: given a
target pixel delta, the inverse of the forward model says
"send these HID dxs". When prediction error exceeds the click
tolerance, the homer falls back to its existing closed-loop
behaviour — so the model can never make things worse than the
status quo, only faster.

The forward model is light enough (~600 weights) that we
implement it in pure NumPy at inference time, avoiding an MLX
import on the homer's hot path. Training still uses MLX (faster).

Inverse: Newton's method on the forward model. We initialise
with a linear-extrapolation guess (the ratio learned online by
the homer would also work) and iterate 3-5 times — small dim,
small dataset, converges fast.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PointerAccelConfig:
    hidden: int
    input_features: list[str]
    output_features: list[str]
    platform: str = "ubuntu-libinput-adaptive"
    # "forward" → (hid, cursor) → measured; runtime must Newton-invert.
    # "inverse" → (measured, cursor) → hid; runtime is a single fwd pass.
    direction: str = "forward"


class PointerAccelModel:
    """Forward + inverse wrapper around a trained MLP."""

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = Path(weights_dir)
        cfg_path = self.weights_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"pointer-accel config missing at {cfg_path}"
            )
        cfg = json.loads(cfg_path.read_text("utf-8"))
        self.config = PointerAccelConfig(
            hidden=int(cfg["hidden"]),
            input_features=list(cfg.get("input_features", [])),
            output_features=list(cfg.get("output_features", [])),
            platform=str(cfg.get("platform", "")),
            direction=str(cfg.get("direction", "forward")),
        )
        npz = np.load(self.weights_dir / "weights.npz")
        self._w1 = npz["fc1.weight"].astype(np.float32)
        self._b1 = npz["fc1.bias"].astype(np.float32)
        self._w2 = npz["fc2.weight"].astype(np.float32)
        self._b2 = npz["fc2.bias"].astype(np.float32)
        self._w3 = npz["fc3.weight"].astype(np.float32)
        self._b3 = npz["fc3.bias"].astype(np.float32)
        logger.info(
            "PointerAccelModel: loaded %d-hidden MLP from %s "
            "(platform=%s)",
            self.config.hidden, self.weights_dir, self.config.platform,
        )

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        # Tanh approximation matches MLX/torch's default.
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """``x`` shape (4,) or (N, 4)."""
        was_1d = x.ndim == 1
        if was_1d:
            x = x.reshape(1, 4)
        h = self._gelu(x @ self._w1.T + self._b1)
        h = self._gelu(h @ self._w2.T + self._b2)
        out = h @ self._w3.T + self._b3
        return out[0] if was_1d else out

    def predict(
        self, hid_dx: int, hid_dy: int,
        cursor_x_pct: float, cursor_y_pct: float,
    ) -> tuple[float, float]:
        """Forward: given an HID delta + current cursor position,
        return the predicted observed cursor delta in normalised
        image coordinates."""
        x = np.array([
            hid_dx / 127.0,
            hid_dy / 127.0,
            cursor_x_pct * 2.0 - 1.0,
            cursor_y_pct * 2.0 - 1.0,
        ], dtype=np.float32)
        out = self._forward(x)
        return float(out[0]), float(out[1])

    def inverse(
        self,
        target_dx_pct: float, target_dy_pct: float,
        cursor_x_pct: float, cursor_y_pct: float,
        *,
        initial_ratio_x: float | None = None,
        initial_ratio_y: float | None = None,
        max_iters: int = 6,
        tol_pct: float = 0.002,
    ) -> tuple[int, int]:
        """Given a desired observed cursor delta, return the HID
        (dx, dy) that should produce it under the trained model.

        If the checkpoint was trained in ``direction="inverse"`` (the
        default in v3+), this is a single forward pass — no Newton
        iteration, no Jacobian noise, much cleaner near zero. If the
        checkpoint is a legacy forward model, we Newton-invert.
        """
        if self.config.direction == "inverse":
            x = np.array([
                float(target_dx_pct),
                float(target_dy_pct),
                cursor_x_pct * 2.0 - 1.0,
                cursor_y_pct * 2.0 - 1.0,
            ], dtype=np.float32)
            out = self._forward(x)
            # Outputs are HID normalised by /127; un-normalise and
            # clamp to the signed-byte range the BT report supports.
            dx = max(-127.0, min(127.0, float(out[0]) * 127.0))
            dy = max(-127.0, min(127.0, float(out[1]) * 127.0))
            return int(round(dx)), int(round(dy))
        # Legacy forward-model path — Newton's method on a 2-d
        # problem with finite-difference Jacobian.
        # Seed: dx ≈ target_dx / ratio_x. Clamp to ±127 right away.
        rx = initial_ratio_x if initial_ratio_x and initial_ratio_x > 0 else 1.0 / 40.0
        ry = initial_ratio_y if initial_ratio_y and initial_ratio_y > 0 else 1.0 / 40.0
        dx = float(target_dx_pct / rx) if rx else 0.0
        dy = float(target_dy_pct / ry) if ry else 0.0
        dx = max(-127.0, min(127.0, dx))
        dy = max(-127.0, min(127.0, dy))

        # Numerical-Jacobian Newton step. Step size is small but
        # bigger than the model's own resolution at the seed point.
        eps = 1.0
        target = np.array([target_dx_pct, target_dy_pct], dtype=np.float32)
        for _ in range(max_iters):
            f0 = np.array(
                self.predict(int(round(dx)), int(round(dy)),
                             cursor_x_pct, cursor_y_pct),
                dtype=np.float32,
            )
            err = target - f0
            if abs(float(err[0])) < tol_pct and abs(float(err[1])) < tol_pct:
                break
            # ∂f/∂dx via forward differences
            f_dx = np.array(
                self.predict(int(round(dx + eps)), int(round(dy)),
                             cursor_x_pct, cursor_y_pct),
                dtype=np.float32,
            )
            f_dy = np.array(
                self.predict(int(round(dx)), int(round(dy + eps)),
                             cursor_x_pct, cursor_y_pct),
                dtype=np.float32,
            )
            J = np.column_stack([(f_dx - f0) / eps, (f_dy - f0) / eps])
            try:
                step = np.linalg.solve(J, err)
            except np.linalg.LinAlgError:
                # Singular — bail with current best.
                break
            dx = max(-127.0, min(127.0, dx + float(step[0])))
            dy = max(-127.0, min(127.0, dy + float(step[1])))
        return int(round(dx)), int(round(dy))
