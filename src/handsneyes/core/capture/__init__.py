"""Capture: abstract source + webcam backend."""

from handsneyes.core.capture.base import (
    CapturedFrame,
    CaptureError,
    CaptureSource,
    CropRegion,
)
from handsneyes.core.capture.webcam import WebcamCapture

__all__ = [
    "CapturedFrame",
    "CaptureError",
    "CaptureSource",
    "CropRegion",
    "WebcamCapture",
]
