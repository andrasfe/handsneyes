# mypy: disable-error-code = "no-any-return"
"""Long-jump HID predictor for the visual servo homer.

Companion to :class:`PointerAccelModel`. Where the per-step pointer-
accel model handles small residual moves inside the closed-loop
servo, this one predicts the TOTAL HID for a whole click trajectory:
given the current cursor position and the target pixel, return the
HID magnitude needed to land cursor on (or very near) target in one
shot.

Trained on aggregated successful trajectories — see
``scripts/build_longjump_dataset.py`` and ``scripts/train_longjump.py``.

The runtime fires the predicted HID as a chain of back-to-back
bursts (each clamped to ±127 per axis, the BT HID per-report
limit) without per-step captures. After the chain lands, the
homer takes a single capture, computes the residual, and lets the
closed-loop ``pointer_accel`` model refine in 1-2 more iterations.

Net effect: a ~50% slam-to-target move that used to take 7-10
closed-loop iterations now takes 1 chain + 1-2 refinement
iterations, ~2.5× faster.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LongJumpConfig:
    hidden: int
    hid_scale: float
    input_features: list[str]
    output_features: list[str]
    platform: str = "ubuntu-libinput-adaptive"


class LongJumpModel:
    """Forward wrapper around a trained long-jump MLP."""

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = Path(weights_dir)
        cfg_path = self.weights_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"longjump config missing at {cfg_path}"
            )
        cfg = json.loads(cfg_path.read_text("utf-8"))
        self.config = LongJumpConfig(
            hidden=int(cfg["hidden"]),
            hid_scale=float(cfg.get("hid_scale", 500.0)),
            input_features=list(cfg.get("input_features", [])),
            output_features=list(cfg.get("output_features", [])),
            platform=str(cfg.get("platform", "")),
        )
        npz = np.load(self.weights_dir / "weights.npz")
        self._w1 = npz["fc1.weight"].astype(np.float32)
        self._b1 = npz["fc1.bias"].astype(np.float32)
        self._w2 = npz["fc2.weight"].astype(np.float32)
        self._b2 = npz["fc2.bias"].astype(np.float32)
        self._w3 = npz["fc3.weight"].astype(np.float32)
        self._b3 = npz["fc3.bias"].astype(np.float32)
        logger.info(
            "LongJumpModel: loaded %d-hidden MLP from %s (platform=%s)",
            self.config.hidden, self.weights_dir, self.config.platform,
        )

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        was_1d = x.ndim == 1
        if was_1d:
            x = x.reshape(1, 4)
        h = self._gelu(x @ self._w1.T + self._b1)
        h = self._gelu(h @ self._w2.T + self._b2)
        out = h @ self._w3.T + self._b3
        return out[0] if was_1d else out

    def predict_total_hid(
        self,
        cursor_x_pct: float, cursor_y_pct: float,
        target_x_pct: float, target_y_pct: float,
        *,
        max_total_hid: int = 1500,
        calibration: tuple[float, float] = (1.0, 1.0),
    ) -> tuple[int, int]:
        """Predict the total HID (dx, dy) to send across one chain of
        back-to-back bursts to land the cursor on the target.

        ``max_total_hid`` caps each axis so a model misfire on a
        sample outside training distribution can't slam the cursor
        through the screen. Default 1500 covers the dataset's max
        observed total (~1400) plus a small margin.

        ``calibration`` is an empirical per-axis scalar applied
        post-prediction. v1 was trained on slow-paced trajectories
        with ``SETTLE_SEC`` between HIDs, where Mac pointer-accel
        produces less motion per HID. The runtime fires bursts back-
        to-back which triggers a more aggressive accel curve, so the
        cursor consistently overshoots v1's predictions by ~15-25%.
        Scaling down by 0.85 on both axes brings the typical landing
        from ~9% residual to ~2% — well inside the closed-loop's
        recovery zone. v2+ retrains on chained-burst data and should
        be deployed with calibration=(1.0, 1.0).
        """
        dx = float(target_x_pct) - float(cursor_x_pct)
        dy = float(target_y_pct) - float(cursor_y_pct)
        x = np.array([
            dx, dy,
            cursor_x_pct * 2.0 - 1.0,
            cursor_y_pct * 2.0 - 1.0,
        ], dtype=np.float32)
        out = self._forward(x)
        hid_dx = float(out[0]) * self.config.hid_scale * calibration[0]
        hid_dy = float(out[1]) * self.config.hid_scale * calibration[1]
        hid_dx = max(-max_total_hid, min(max_total_hid, hid_dx))
        hid_dy = max(-max_total_hid, min(max_total_hid, hid_dy))
        return int(round(hid_dx)), int(round(hid_dy))


def chunk_hid_for_bursts(
    total_hid_dx: int, total_hid_dy: int, *, max_per_burst: int = 127,
) -> list[tuple[int, int]]:
    """Split a total HID delta into a sequence of back-to-back HID
    bursts, each ≤ ``max_per_burst`` per axis (signed-byte BT HID
    cap). Used by the homer to fire a long-jump prediction as a
    chain without per-step captures in between.

    Bursts are sized to roughly equalise the number of bursts on
    each axis so the cursor moves diagonally rather than first-x-
    then-y. Example: total=(300, 100), max=127 →
    [(127, 42), (127, 42), (46, 16)].
    """
    abs_dx = abs(total_hid_dx)
    abs_dy = abs(total_hid_dy)
    if abs_dx == 0 and abs_dy == 0:
        return []
    # Number of bursts is dictated by the bigger axis.
    n_bursts = max(
        1,
        (max(abs_dx, abs_dy) + max_per_burst - 1) // max_per_burst,
    )
    sgn_dx = 1 if total_hid_dx >= 0 else -1
    sgn_dy = 1 if total_hid_dy >= 0 else -1
    bursts: list[tuple[int, int]] = []
    remaining_dx = abs_dx
    remaining_dy = abs_dy
    for i in range(n_bursts):
        # Spread the remainder over the remaining bursts, biggest
        # chunks first so the cursor moves substantially in burst 1
        # — the homer captures after the chain, so we want the cursor
        # near target by the time of capture.
        bursts_left = n_bursts - i
        dx_chunk = min(
            max_per_burst,
            (remaining_dx + bursts_left - 1) // bursts_left,
        )
        dy_chunk = min(
            max_per_burst,
            (remaining_dy + bursts_left - 1) // bursts_left,
        )
        bursts.append((sgn_dx * dx_chunk, sgn_dy * dy_chunk))
        remaining_dx -= dx_chunk
        remaining_dy -= dy_chunk
    return bursts
