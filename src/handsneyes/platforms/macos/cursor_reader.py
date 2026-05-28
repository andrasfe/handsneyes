"""QuartzCursorReader — macOS-native cursor position oracle.

``Quartz.CGEventGetLocation`` returns the cursor position with pixel
precision and zero observable latency. This is the only path that
works on macOS self-capture, where the cursor isn't composited into
the framebuffer that ``screencapture`` reads.

The reader normalises by the main display's pixel size — matches the
homer's convention that all coordinates are fractions of the active
target's screen.

Import is lazy: the ``Quartz`` package only exists when
``pyobjc-framework-Quartz`` is installed (mac-only). Modules that
construct this reader on a non-mac host get a clean ``ImportError``
that the factory turns into a ``cursor_reader=None`` context.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class QuartzCursorReader:
    """Reads the cursor location via Core Graphics."""

    def __init__(self) -> None:
        import Quartz  # noqa: F401 — surfaces the missing dep early

        self._Quartz = Quartz
        main = Quartz.CGMainDisplayID()
        self._sw = int(Quartz.CGDisplayPixelsWide(main))
        self._sh = int(Quartz.CGDisplayPixelsHigh(main))
        logger.info(
            "QuartzCursorReader online — main display %dx%d",
            self._sw, self._sh,
        )

    async def read_pct(self) -> tuple[float, float] | None:
        try:
            e = self._Quartz.CGEventCreate(None)
            p = self._Quartz.CGEventGetLocation(e)
            x = float(p.x) / self._sw
            y = float(p.y) / self._sh
            # Clamp to a slightly-wider range so a cursor sitting one
            # pixel past the display edge still reports a sane number.
            return (max(-0.05, min(1.05, x)), max(-0.05, min(1.05, y)))
        except Exception as e:
            logger.warning("Quartz cursor read failed: %s", e)
            return None
