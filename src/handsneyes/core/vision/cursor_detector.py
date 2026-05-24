"""Single-frame cursor detector (numpy-only inference).

Replaces the homer's 6-frame oscillation-variance loop with one
forward pass over a single webcam frame. Faster (1 capture vs 7) and
in principle sub-pixel accurate when well-trained.

The architecture mirrors ``scripts/train_cursor_detector.py``: a tiny
4-conv CNN producing a 96x54 heatmap, then a DSNT spatial-softmax
readout to recover (x_pct, y_pct).

Inference is pure numpy at the request of the homer's hot path —
avoiding an MLX import keeps the click_at latency budget cleaner.
Training stays on MLX (much faster).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CursorDetectorConfig:
    in_h: int
    in_w: int
    head_h: int
    head_w: int
    platform: str = ""
    best_val_mse: float = 0.0


class CursorDetector:
    """Numpy-only forward pass for the trained cursor detector."""

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = Path(weights_dir)
        cfg_path = self.weights_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"cursor-detector config missing at {cfg_path}"
            )
        cfg = json.loads(cfg_path.read_text("utf-8"))
        self.config = CursorDetectorConfig(
            in_h=int(cfg["in_h"]), in_w=int(cfg["in_w"]),
            head_h=int(cfg["head_h"]), head_w=int(cfg["head_w"]),
            platform=str(cfg.get("platform", "")),
            best_val_mse=float(cfg.get("best_val_mse", 0.0)),
        )
        npz = np.load(self.weights_dir / "weights.npz")
        # MLX nn.Conv2d uses (out_c, kh, kw, in_c) weight layout. We
        # keep that for storage and transpose at inference time as
        # needed by our hand-rolled conv (which uses scipy/np patches).
        self._params = {k: npz[k].astype(np.float32) for k in npz.files}
        logger.info(
            "CursorDetector: loaded from %s (in=%dx%d head=%dx%d)",
            self.weights_dir, self.config.in_h, self.config.in_w,
            self.config.head_h, self.config.head_w,
        )

        # Pre-compute the DSNT coordinate grids.
        self._gx = np.linspace(
            0, 1, self.config.head_w, dtype=np.float32,
        )
        self._gy = np.linspace(
            0, 1, self.config.head_h, dtype=np.float32,
        )

    # ── numpy primitives ──────────────────────────────────────────

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _conv2d(
        x: np.ndarray, w: np.ndarray, b: np.ndarray,
        *, stride: int, padding: int,
    ) -> np.ndarray:
        """NHWC convolution via cv2.filter2D per channel… no, via
        im2col. For our small feature maps this is fast enough.

        x: (N, H, W, Cin)
        w: (Cout, KH, KW, Cin) — MLX layout
        b: (Cout,)
        """
        N, H, W, Cin = x.shape
        Cout, KH, KW, _ = w.shape
        if padding > 0:
            x = np.pad(
                x,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="constant",
            )
        H_out = (H + 2 * padding - KH) // stride + 1
        W_out = (W + 2 * padding - KW) // stride + 1
        # im2col: (N, H_out, W_out, KH*KW*Cin)
        cols = np.empty(
            (N, H_out, W_out, KH * KW * Cin), dtype=np.float32,
        )
        for i in range(KH):
            for j in range(KW):
                src = x[
                    :,
                    i: i + stride * H_out: stride,
                    j: j + stride * W_out: stride,
                    :,
                ]
                cols[..., (i * KW + j) * Cin: (i * KW + j + 1) * Cin] = src
        # Weight reshape: (Cout, KH*KW*Cin)
        w_flat = w.reshape(Cout, KH * KW * Cin)
        out = cols @ w_flat.T + b
        return out  # (N, H_out, W_out, Cout)

    # ── forward pass ──────────────────────────────────────────────

    def _heatmap(self, frame_nhwc: np.ndarray) -> np.ndarray:
        """frame: (1, H, W, 1) → heatmap (1, head_h, head_w, 1)."""
        p = self._params
        x = self._conv2d(
            frame_nhwc, p["c1.weight"], p["c1.bias"],
            stride=2, padding=2,
        )
        x = self._gelu(x)
        x = self._conv2d(
            x, p["c2.weight"], p["c2.bias"], stride=2, padding=1,
        )
        x = self._gelu(x)
        x = self._conv2d(
            x, p["c3.weight"], p["c3.bias"], stride=1, padding=1,
        )
        x = self._gelu(x)
        x = self._conv2d(
            x, p["c4.weight"], p["c4.bias"], stride=1, padding=1,
        )
        x = self._gelu(x)
        x = self._conv2d(
            x, p["head.weight"], p["head.bias"], stride=1, padding=0,
        )
        return x

    def predict(
        self, image_bgr_or_gray: np.ndarray,
    ) -> tuple[float, float] | None:
        """Locate the cursor in a single webcam frame.

        Returns (x_pct, y_pct) in normalised image coordinates, or
        None if the input frame is empty/degenerate.
        """
        if image_bgr_or_gray is None or image_bgr_or_gray.size == 0:
            return None
        if image_bgr_or_gray.ndim == 3:
            gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr_or_gray
        resized = cv2.resize(
            gray, (self.config.in_w, self.config.in_h),
            interpolation=cv2.INTER_AREA,
        )
        x = (resized.astype(np.float32) / 255.0)[None, ..., None]  # (1,H,W,1)
        heatmap = self._heatmap(x)  # (1, h, w, 1)
        h = heatmap[0, ..., 0]  # (h, w)
        # Softmax over the whole heatmap, then expected coords.
        h_flat = h.reshape(-1)
        h_flat = h_flat - h_flat.max()  # stability
        p = np.exp(h_flat)
        p = p / p.sum()
        p = p.reshape(self.config.head_h, self.config.head_w)
        px = p.sum(axis=0)  # (W,)
        py = p.sum(axis=1)  # (H,)
        x_pct = float((px * self._gx).sum())
        y_pct = float((py * self._gy).sum())
        return x_pct, y_pct
