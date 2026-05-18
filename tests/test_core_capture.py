"""Tests for handsneyes.core.capture."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    import pytest_mock

from handsneyes.core.capture import (
    CapturedFrame,
    CaptureError,
    CaptureSource,
    CropRegion,
    WebcamCapture,
)


def test_capture_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        CaptureSource()  # type: ignore[abstract]


def test_crop_region_is_frozen() -> None:
    r = CropRegion(x=0, y=0, width=100, height=50)
    with pytest.raises(ValueError):  # noqa: PT011
        r.x = 10  # type: ignore[misc]


def test_capture_error_raises() -> None:
    with pytest.raises(CaptureError, match="boom"):
        raise CaptureError("boom")


class TestWebcamCapture:
    @pytest.mark.asyncio
    async def test_open_failure_when_device_not_opened(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = False
        mocker.patch(
            "handsneyes.core.capture.webcam.cv2.VideoCapture",
            return_value=fake_cap,
        )
        cam = WebcamCapture(device_index=0)
        with pytest.raises(CaptureError, match="Failed to open"):
            await cam.open()

    @pytest.mark.asyncio
    async def test_capture_frame_returns_ndarray(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        fake_cap.read.return_value = (True, fake_frame)
        fake_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
        mocker.patch(
            "handsneyes.core.capture.webcam.cv2.VideoCapture",
            return_value=fake_cap,
        )
        cam = WebcamCapture(device_index=0)
        await cam.open()
        frame = await cam.capture_frame()
        assert isinstance(frame, CapturedFrame)
        assert frame.image.shape == (480, 640, 3)
        assert frame.frame_number == 1
        assert frame.source_device == "webcam:0"
        await cam.close()

    @pytest.mark.asyncio
    async def test_capture_frame_raises_when_not_open(self) -> None:
        cam = WebcamCapture(device_index=0)
        with pytest.raises(CaptureError, match="not open"):
            await cam.capture_frame()
