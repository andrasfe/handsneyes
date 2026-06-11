"""RectifiedCapture — perspective-corrected wrapper for webcam sources.

Decorates any :class:`CaptureSource`: on ``open()`` it grabs one
frame, detects the target screen's quadrilateral, and freezes the
homography for the session; every subsequent ``capture_frame()``
returns the frame warped into true screen space.

Failure is never fatal — when no screen quad is detectable (display
asleep, camera mispointed), it falls back to a cached calibration
from a previous session, and failing that passes frames through
unmodified so the run degrades to today's unrectified behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

from handsneyes.core.capture.base import CapturedFrame, CaptureSource
from handsneyes.core.vision.screen_geometry import (
    ScreenRectifier,
    detect_screen_quad,
)

logger = logging.getLogger(__name__)


class RectifiedCapture(CaptureSource):
    def __init__(
        self,
        inner: CaptureSource,
        *,
        aspect_hint: tuple[int, int] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._aspect_hint = aspect_hint
        self._cache_path = cache_path
        self._rectifier: ScreenRectifier | None = None

    @property
    def rectifier(self) -> ScreenRectifier | None:
        return self._rectifier

    async def open(self) -> None:
        await self._inner.open()
        self._is_open = True
        await self._calibrate()

    async def close(self) -> None:
        self._is_open = False
        await self._inner.close()

    async def _calibrate(self) -> None:
        try:
            frame = await self._inner.capture_frame()
        except Exception as e:  # noqa: BLE001
            logger.warning("rectify: calibration capture failed: %s", e)
            return
        img = frame.image
        fh, fw = img.shape[:2]
        quad = detect_screen_quad(img)
        if quad is not None:
            self._rectifier = ScreenRectifier.from_quad(
                quad, (fw, fh), aspect_hint=self._aspect_hint,
            )
            logger.info(
                "rectify: screen quad found (coverage %.0f%%) → "
                "warping to %dx%d",
                self._rectifier.coverage * 100,
                *self._rectifier.out_size,
            )
            if self._cache_path is not None:
                try:
                    self._rectifier.save(self._cache_path)
                except Exception as e:  # noqa: BLE001
                    logger.debug("rectify: cache save failed: %s", e)
            return
        if self._cache_path is not None:
            cached = ScreenRectifier.load(
                self._cache_path, expect_frame_size=(fw, fh),
            )
            if cached is not None:
                self._rectifier = cached
                logger.info(
                    "rectify: no quad in current frame — using cached "
                    "calibration from %s", self._cache_path,
                )
                return
        logger.warning(
            "rectify: no screen quad detected and no usable cache — "
            "running unrectified",
        )

    async def capture_frame(self) -> CapturedFrame:
        frame = await self._inner.capture_frame()
        if self._rectifier is None:
            return frame
        return frame.model_copy(
            update={"image": self._rectifier.rectify(frame.image)},
        )
