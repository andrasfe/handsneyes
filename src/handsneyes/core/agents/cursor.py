"""CursorAgent — locate the mouse cursor in a captured frame.

Three detection paths (cascade):

  1. **HSV** — fast, zero model cost, works when the target machine
     uses a high-contrast cursor theme (e.g. Ubuntu's ``redglass`` at
     size 96). Motion-verifies the candidate by sending a known nudge
     to reject static red UI elements.
  2. **Oscillation-variance** — robust on any default cursor. Sends
     a short alternating jiggle pattern, captures frames during, and
     finds the pixel cluster with highest std-dev across them.
  3. **ROI-prior diff** — used per-step during a servo loop. Given
     the previous cursor position and an expected delta, diff
     pre/post frames inside an ROI around the expected new position
     and pick the largest changed blob.

Reuses helpers in :mod:`handsneyes.core.vision.cursor_finder` so this
agent is a thin wrapper — no behavioural change.
"""

# ruff: noqa: N806  # M = cv2.moments(...), mathematical convention

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np  # noqa: TC002 — used by signatures inside the class body

from handsneyes.core.agents.base import Agent, Outcome
from handsneyes.core.vision.cursor_finder import (
    CursorHit,
    find_cursor_by_variance,
    find_cursor_hsv,
)

logger = logging.getLogger(__name__)


@dataclass
class CursorOutcome(Outcome):
    """``data['position'] = (x_pct, y_pct)`` on success."""


class CursorAgent(Agent):
    """Locate the cursor in image-fraction coordinates."""

    name = "cursor"

    async def run(  # type: ignore[override]
        self,
        *,
        mode: str = "auto",
        image: np.ndarray | None = None,
        verify_motion: bool = True,
    ) -> CursorOutcome:
        """Find the cursor.

        ``mode``:
          - ``"auto"`` (default): HSV → fall back to oscillation
          - ``"hsv"``:   HSV only on the supplied (or freshly captured)
                         image. Optional motion verification.
          - ``"oscillation"``: jiggle + variance detection.
          - ``"diff"``:  use :meth:`diff_with_prior` instead — call
                         that method directly, ``run`` doesn't expose it.
        """
        if self.ctx.capture is None and image is None:
            return CursorOutcome(
                success=False, reason="no capture in context",
            )

        if mode in ("auto", "hsv"):
            frame = image if image is not None else (
                await self._capture_color()
            )
            hit = find_cursor_hsv(frame)
            if hit is not None:
                if verify_motion:
                    verified = await self._verify_hsv_by_motion(
                        (hit.x_pct, hit.y_pct),
                    )
                    if verified is not None:
                        return CursorOutcome(
                            success=True,
                            reason="HSV (motion-verified)",
                            data={
                                "position": verified,
                                "method": "hsv",
                                "area_pct": hit.area_pct,
                                "confidence": hit.confidence,
                            },
                        )
                else:
                    return CursorOutcome(
                        success=True,
                        reason="HSV (unverified)",
                        data={
                            "position": (hit.x_pct, hit.y_pct),
                            "method": "hsv_unverified",
                            "area_pct": hit.area_pct,
                            "confidence": hit.confidence,
                        },
                    )
            if mode == "hsv":
                return CursorOutcome(
                    success=False, reason="HSV did not find a cursor",
                )

        if mode in ("auto", "oscillation"):
            pos = await self.find_via_oscillation()
            if pos is not None:
                return CursorOutcome(
                    success=True,
                    reason="oscillation-variance",
                    data={"position": pos, "method": "oscillation"},
                )
            return CursorOutcome(
                success=False,
                reason="oscillation could not detect a cursor",
            )

        return CursorOutcome(
            success=False, reason=f"unknown mode {mode!r}",
        )

    # ───────────────────── individual finders ─────────────────────

    async def find_via_hsv(
        self, image: np.ndarray | None = None,
    ) -> CursorHit | None:
        """Run HSV detection on a frame; no motion verification."""
        if image is None:
            image = await self._capture_color()
        return find_cursor_hsv(image)

    async def find_via_oscillation(self) -> tuple[float, float] | None:
        """Send a short jiggle pattern and find the moving cluster."""
        if self.ctx.mouse is None or self.ctx.capture is None:
            return None
        frames: list[np.ndarray] = []
        frames.append(await self._capture_gray())
        # Small symmetric oscillation — trajectory centres on start.
        oscillation = [
            (20, 0), (-40, 0), (40, 0),
            (0, 20), (0, -40), (0, 40),
        ]
        for dx, dy in oscillation:
            try:
                await self.ctx.mouse.move(dx, dy)
            except Exception as e:
                logger.warning("Oscillation move failed: %s", e)
                return None
            await asyncio.sleep(0.10)
            frames.append(await self._capture_gray())
        return find_cursor_by_variance(frames)

    async def diff_with_prior(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        prev_cursor: tuple[float, float],
        expected_new: tuple[float, float],
        *,
        roi_radius: float = 0.20,
    ) -> tuple[float, float] | None:
        """Locate cursor in ``post`` by diffing against ``pre`` inside an
        ROI around ``expected_new``. Picks the blob whose centroid score
        (distance-to-expected minus 0.3·distance-to-prev) is lowest.
        """
        if pre.ndim == 3:
            pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        if post.ndim == 3:
            post = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
        h, w = pre.shape[:2]

        ex, ey = expected_new
        cx, cy = prev_cursor
        x0 = int(max(0, min(ex, cx) - roi_radius) * w)
        y0 = int(max(0, min(ey, cy) - roi_radius) * h)
        x1 = int(min(1, max(ex, cx) + roi_radius) * w)
        y1 = int(min(1, max(ey, cy) + roi_radius) * h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None

        diff = cv2.absdiff(pre[y0:y1, x0:x1], post[y0:y1, x0:x1])
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        roi_area = (x1 - x0) * (y1 - y0)
        ex_px, ey_px = ex * w, ey * h
        cx_px, cy_px = cx * w, cy * h
        best: tuple[float, float] | None = None
        best_d = float("inf")
        for c in contours:
            area = cv2.contourArea(c)
            if area < roi_area * 0.0005 or area > roi_area * 0.10:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            blob_x = M["m10"] / M["m00"] + x0
            blob_y = M["m01"] / M["m00"] + y0
            d_new = math.hypot(blob_x - ex_px, blob_y - ey_px)
            d_old = math.hypot(blob_x - cx_px, blob_y - cy_px)
            score = d_new - 0.3 * d_old
            if score < best_d:
                best_d = score
                best = (blob_x / w, blob_y / h)
        return best

    # ───────────────────── helpers ─────────────────────

    async def _verify_hsv_by_motion(
        self, candidate: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Send a known nudge and confirm the candidate moved."""
        if self.ctx.mouse is None or self.ctx.capture is None:
            return None
        nudge_hid = 80
        try:
            await self.ctx.mouse.move(nudge_hid, 0)
        except Exception:
            return None
        await asyncio.sleep(0.30)
        post = await self._capture_color()
        new_hit = find_cursor_hsv(post)
        if new_hit is None:
            return None
        observed_dx = new_hit.x_pct - candidate[0]
        observed_dy = new_hit.y_pct - candidate[1]
        # Expected ~ 80 HID * 1.6/1920 = 0.067; require ≥30% of that.
        if observed_dx < 0.020:
            return None
        if abs(observed_dy) > 0.05:
            return None
        return (new_hit.x_pct, new_hit.y_pct)

    async def _capture_gray(self) -> np.ndarray:
        assert self.ctx.capture is not None
        frame = await self.ctx.capture.capture_frame()
        return cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

    async def _capture_color(self) -> np.ndarray:
        assert self.ctx.capture is not None
        frame = await self.ctx.capture.capture_frame()
        return frame.image
