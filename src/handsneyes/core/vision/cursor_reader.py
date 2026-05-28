"""CursorReader — direct cursor-position oracle.

The visual-servo homer normally finds the cursor by analysing the
captured frame (HSV mask, oscillation variance, frame diff). That
works whenever the cursor is *in* the captured frame — webcams pointed
at a target machine always include the cursor as photons leaving the
screen.

It breaks on macOS self-capture: ``screencapture`` and PIL ``ImageGrab``
both read from a framebuffer that the OS composites *without* the
cursor (cursor lives in a separate hardware overlay). The frame is
correct; the cursor is just nowhere in it. No HSV mask or frame-diff
will recover what isn't there.

A ``CursorReader`` is a per-target oracle that knows where the cursor
actually is, bypassing the visual loop. The macOS implementation calls
``Quartz.CGEventGetLocation``. Linux/X11 could implement one via
``xdotool getmouselocation``; Wayland has no equivalent and falls back
to the visual path.

Returning ``None`` means "no fast path; use the visual loop." Agents
should always treat the reader as a fast hint and keep the visual
loop as the durable mechanism.
"""

from __future__ import annotations

from typing import Protocol


class CursorReader(Protocol):
    """Returns the cursor position as ``(x_pct, y_pct)`` in [0, 1].

    Implementations should be cheap (microseconds) and may raise on
    unexpected platform errors — callers wrap in try/except and fall
    back to the visual finder when this returns ``None`` or throws.
    """

    async def read_pct(self) -> tuple[float, float] | None:
        ...
