"""Webcam capture implementation using OpenCV.

Runs OpenCV's blocking capture in a thread pool executor to avoid
blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import cv2
import numpy as np  # noqa: TC002  (used as type annotation on private helpers)

from handsneyes.core.capture.base import (
    CapturedFrame,
    CaptureError,
    CaptureSource,
    CropRegion,
)

logger = logging.getLogger(__name__)


class WebcamCapture(CaptureSource):
    """Captures frames from a webcam using OpenCV."""

    def __init__(
        self,
        device_index: int = 0,
        crop_region: CropRegion | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(crop_region=crop_region)
        self._device_index = device_index
        self._resolution = resolution
        self._cap: cv2.VideoCapture | None = None
        self._mirror = False

    async def open(self) -> None:
        """Open the capture device."""
        loop = asyncio.get_event_loop()
        self._cap = await loop.run_in_executor(
            None, cv2.VideoCapture, self._device_index
        )
        if not self._cap.isOpened():
            raise CaptureError(
                f"Failed to open capture device {self._device_index}"
            )
        if self._resolution:
            w, h = self._resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._is_open = True
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read several frames to let autofocus settle
        for _ in range(20):
            await loop.run_in_executor(None, self._cap.read)
            await asyncio.sleep(0.03)

        # Photo Booth mirrors for preview, but OpenCV gets raw data.
        # Set via config if a specific device returns mirrored images.
        self._mirror = False

        logger.info(
            "Opened capture device %d (%dx%d)",
            self._device_index,
            actual_w,
            actual_h,
        )

    async def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug(
                "Released capture device %d", self._device_index
            )
        self._cap = None
        self._is_open = False

    async def capture_frame(self) -> CapturedFrame:
        """Capture a single frame from the webcam."""
        if not self._is_open or self._cap is None:
            raise CaptureError("Webcam is not open")
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self._capture_sync)
        if self._crop_region is not None:
            frame = self._apply_crop(frame)
        self._frame_counter += 1
        return CapturedFrame(
            image=frame,
            timestamp=datetime.now(),
            frame_number=self._frame_counter,
            source_device=f"webcam:{self._device_index}",
            crop_applied=self._crop_region,
        )

    def _capture_sync(self) -> np.ndarray:
        """Synchronous frame capture (runs in thread pool)."""
        assert self._cap is not None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CaptureError("Failed to read frame from webcam")
        if self._mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        """Apply crop region to a frame."""
        r = self._crop_region
        assert r is not None
        return frame[r.y : r.y + r.height, r.x : r.x + r.width].copy()
