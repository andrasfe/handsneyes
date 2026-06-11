# ruff: noqa: N806
"""Screen-quad detection + perspective rectification.

A webcam never views the target screen dead-on: every degree of
off-axis viewing skews the pct→pixel mapping that the homer, OCR,
and snap finders all assume is linear. This module finds the
screen's quadrilateral in the webcam frame (the screen is the
dominant bright rectangle in an indoor scene), computes the
homography that maps it to a true rectangle, and warps frames so
all downstream vision operates in screen space.

Purely passive — nothing is installed or displayed on the target.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# A real screen fills a meaningful part of the camera view; tiny
# bright quads are windows / lamps / monitors in the background.
MIN_COVERAGE = 0.15
# A quad covering ~the whole frame is the camera staring at the
# screen point-blank (or a white wall) — rectifying it is a no-op
# at best and an artefact amplifier at worst.
MAX_COVERAGE = 0.98


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]   # TL: smallest x+y
    out[2] = pts[np.argmax(s)]   # BR: largest x+y
    out[1] = pts[np.argmin(d)]   # TR: smallest y-x
    out[3] = pts[np.argmax(d)]   # BL: largest y-x
    return out


def _quad_from_contour(
    contour: np.ndarray, img_area: float,
) -> np.ndarray | None:
    area = cv2.contourArea(contour)
    if not (MIN_COVERAGE * img_area <= area <= MAX_COVERAGE * img_area):
        return None
    peri = cv2.arcLength(contour, True)
    # Sweep epsilon: a screen with a rounded bezel or noisy edges
    # may need a coarser approximation before it collapses to 4 pts.
    for eps in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = order_corners(approx.reshape(4, 2))
            # Degenerate-quad guard: every side meaningfully long.
            sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if sides.min() > np.sqrt(img_area) * 0.10:
                return quad
    return None


def detect_screen_quad(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Find the target screen's 4 corners in a webcam frame.

    Returns a (4, 2) float32 array ordered TL, TR, BR, BL (pixel
    coords), or ``None`` when no plausible screen quad is present
    (screen asleep, camera pointing elsewhere, room brighter than
    the display).

    Two-strategy cascade:
      1. Brightness: the lit screen is the dominant bright region
         indoors. Otsu-threshold + a hard close (fills dark page
         content inside the screen) → largest contour → 4-corner
         approx.
      2. Edges: when screen content is dark enough that Otsu splits
         it, the bezel-to-screen boundary still produces a strong
         edge loop. Canny + dilate → same quad filter.
    """
    if frame_bgr is None or frame_bgr.ndim not in (2, 3):
        return None
    gray = (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if frame_bgr.ndim == 3 else frame_bgr
    )
    h, w = gray.shape[:2]
    img_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strategy 1 — brightness.
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(
        thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        quad = _quad_from_contour(c, img_area)
        if quad is not None:
            return quad

    # Strategy 2 — edges.
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        quad = _quad_from_contour(c, img_area)
        if quad is not None:
            return quad
    return None


class ScreenRectifier:
    """Holds a frozen homography and warps frames into screen space."""

    def __init__(
        self,
        corners: np.ndarray,
        out_size: tuple[int, int],
        frame_size: tuple[int, int],
    ) -> None:
        self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        dst = np.array(
            [
                [0, 0],
                [self.out_size[0] - 1, 0],
                [self.out_size[0] - 1, self.out_size[1] - 1],
                [0, self.out_size[1] - 1],
            ],
            dtype=np.float32,
        )
        self._H = cv2.getPerspectiveTransform(self.corners, dst)

    @classmethod
    def from_quad(
        cls,
        quad: np.ndarray,
        frame_size: tuple[int, int],
        *,
        aspect_hint: tuple[int, int] | None = None,
    ) -> "ScreenRectifier":
        """Build a rectifier from a detected quad.

        Output width comes from the quad's longest horizontal edge
        (preserves the captured resolution — no upsampling). Height
        follows ``aspect_hint`` (the target's true screen aspect,
        e.g. (1920, 1080) from targets.toml) when given, else the
        quad's own measured vertical extent.
        """
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        tl, tr, br, bl = quad
        width = max(
            float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)),
        )
        height = max(
            float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)),
        )
        W = max(2, int(round(width)))
        if aspect_hint is not None and aspect_hint[0] > 0:
            H = max(2, int(round(W * aspect_hint[1] / aspect_hint[0])))
        else:
            H = max(2, int(round(height)))
        return cls(quad, (W, H), frame_size)

    def rectify(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame_bgr, self._H, self.out_size)

    @property
    def coverage(self) -> float:
        quad_area = cv2.contourArea(self.corners.astype(np.float32))
        return float(quad_area) / float(
            self.frame_size[0] * self.frame_size[1],
        )

    # ── persistence ──
    def to_dict(self) -> dict:
        return {
            "corners": self.corners.tolist(),
            "out_size": list(self.out_size),
            "frame_size": list(self.frame_size),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScreenRectifier":
        return cls(
            np.array(d["corners"], dtype=np.float32),
            tuple(d["out_size"]),
            tuple(d["frame_size"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(
        cls, path: Path, expect_frame_size: tuple[int, int] | None = None,
    ) -> "ScreenRectifier | None":
        """Load a persisted calibration. Returns ``None`` (never
        raises) when the file is missing, corrupt, or was calibrated
        at a different capture resolution."""
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            rect = cls.from_dict(d)
        except Exception as e:  # noqa: BLE001
            logger.debug("rectifier load failed (%s): %s", path, e)
            return None
        if (
            expect_frame_size is not None
            and tuple(rect.frame_size) != tuple(expect_frame_size)
        ):
            logger.info(
                "cached rectifier is for %s, capture is %s — ignoring",
                rect.frame_size, expect_frame_size,
            )
            return None
        return rect
