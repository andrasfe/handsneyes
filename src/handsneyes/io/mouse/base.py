"""Abstract base class for mouse action output.

All mouse output backends must conform to this interface, enabling
the system to swap between the HTTP backend (for the Pi REST API)
and potential future backends without changing any other code.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MouseOutput(ABC):
    """Abstract interface for sending mouse actions to a target.

    Implementations translate logical mouse actions (move, click, scroll)
    into the appropriate protocol for their target (HTTP requests to Pi,
    USB HID reports, etc.).

    Example usage::

        async with HttpMouseOutput(base_url="http://10.0.0.2:8080") as mouse:
            await mouse.move(10, -5)
            await mouse.click("left")
            await mouse.scroll(-3)
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the mouse output target.

        Raises:
            MouseOutputError: If connection cannot be established.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection to the output target."""
        ...

    @abstractmethod
    async def move(self, dx: int, dy: int) -> None:
        """Send a relative mouse movement.

        Args:
            dx: Horizontal movement (-127 to 127, positive = right).
            dy: Vertical movement (-127 to 127, positive = down).

        Raises:
            MouseOutputError: If the move cannot be sent.
        """
        ...

    @abstractmethod
    async def click(self, button: str = "left") -> None:
        """Send a mouse button click (press + release).

        Args:
            button: Button to click: "left", "right", or "middle".

        Raises:
            MouseOutputError: If the click cannot be sent.
        """
        ...

    @abstractmethod
    async def scroll(self, amount: int) -> None:
        """Send a scroll wheel action.

        Args:
            amount: Scroll amount (-127 to 127, positive = up).

        Raises:
            MouseOutputError: If the scroll cannot be sent.
        """
        ...

    async def move_to_corner(self, corner: str = "top-left") -> None:
        """Move cursor to a known screen corner by sending large deltas.

        Sends 20 max-delta moves to force the cursor to the specified
        corner, regardless of current position. 20 * 127 = 2540px
        exceeds any reasonable screen dimension.

        Args:
            corner: One of "top-left", "top-right", "bottom-left", "bottom-right".
        """
        dx_sign = -1 if "left" in corner else 1
        dy_sign = -1 if "top" in corner else 1
        for _ in range(20):
            await self.move(dx_sign * 127, dy_sign * 127)
            await asyncio.sleep(0.005)

    async def move_absolute(
        self,
        x_pct: float,
        y_pct: float,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        """Move to an absolute screen position via corner-reset + relative steps.

        First moves to top-left corner, then navigates to the target
        position using relative movements in 127-step chunks.

        Args:
            x_pct: Horizontal position as fraction (0.0 = left, 1.0 = right).
            y_pct: Vertical position as fraction (0.0 = top, 1.0 = bottom).
            screen_width: Target screen width in pixels.
            screen_height: Target screen height in pixels.
        """
        await self.move_to_corner("top-left")

        target_x = int(x_pct * screen_width)
        target_y = int(y_pct * screen_height)
        remaining_x = target_x
        remaining_y = target_y

        while remaining_x > 0 or remaining_y > 0:
            dx = min(remaining_x, 127)
            dy = min(remaining_y, 127)
            if dx > 0 or dy > 0:
                await self.move(dx, dy)
            remaining_x -= dx
            remaining_y -= dy
            await asyncio.sleep(0.005)

    async def click_at(
        self,
        x_pct: float,
        y_pct: float,
        button: str = "left",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        """Move to a screen position and click.

        Args:
            x_pct: Horizontal position as fraction (0.0-1.0).
            y_pct: Vertical position as fraction (0.0-1.0).
            button: Button to click.
            screen_width: Target screen width in pixels.
            screen_height: Target screen height in pixels.
        """
        await self.move_absolute(x_pct, y_pct, screen_width, screen_height)
        await asyncio.sleep(0.05)
        await self.click(button)

    async def __aenter__(self) -> MouseOutput:
        """Async context manager entry -- connects to the output target."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit -- disconnects from the output target."""
        await self.disconnect()


class MouseOutputError(Exception):
    """Raised when mouse output fails."""

    def __init__(self, message: str, backend: str = "") -> None:
        super().__init__(message)
        self.backend = backend
