"""Abstract base class for vision capture sources.

All capture implementations must conform to this interface, enabling
the system to swap between webcam capture, screen capture, or
file-based test sources without changing the rest of the pipeline.

CapturedFrame and CropRegion are colocated here rather than in a
shared domain/ namespace — the capture layer is the only producer
and primary consumer of these types, and we want core/ to stay
namespace-flat.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002  (used by pydantic field type at class creation)
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class CropRegion(BaseModel):
    """Defines a rectangular crop region within a captured frame.

    Coordinates are in pixels, origin at top-left.
    """

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0, description="Left edge x-coordinate in pixels")
    y: int = Field(ge=0, description="Top edge y-coordinate in pixels")
    width: int = Field(gt=0, description="Width of the crop region in pixels")
    height: int = Field(gt=0, description="Height of the crop region in pixels")


class CapturedFrame(BaseModel):
    """A single frame captured from the visual source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(
        description="Raw image data as BGR numpy array (OpenCV format)"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    frame_number: int = Field(ge=0)
    source_device: str = Field(default="webcam")
    crop_applied: CropRegion | None = Field(default=None)


class CaptureError(Exception):
    """Raised when frame capture fails."""


class CaptureSource(ABC):
    """Abstract interface for capturing frames from a visual source."""

    def __init__(self, crop_region: CropRegion | None = None) -> None:
        self._crop_region = crop_region
        self._frame_counter: int = 0
        self._is_open: bool = False

    @property
    def is_open(self) -> bool:
        """Whether the capture device is currently open and ready."""
        return self._is_open

    @abstractmethod
    async def open(self) -> None:
        """Open and initialize the capture device."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release the capture device and free resources."""
        ...

    @abstractmethod
    async def capture_frame(self) -> CapturedFrame:
        """Capture a single frame from the source."""
        ...

    async def stream(
        self, interval: float = 1.0
    ) -> AsyncIterator[CapturedFrame]:
        """Yield frames at the specified interval."""
        import asyncio

        if not self._is_open:
            raise RuntimeError(
                "Capture source is not open. Call open() first."
            )
        while self._is_open:
            frame = await self.capture_frame()
            yield frame
            await asyncio.sleep(interval)

    async def __aenter__(self) -> CaptureSource:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        await self.close()
