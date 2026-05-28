"""ScreenCapture — grab a frame of the local machine's display.

Use case: handsneyes' target IS the same machine running the cc.
The webcam path makes no sense — there's no remote screen to point
a camera at. We just take a screenshot of our own display instead.

Compared to WebcamCapture, this source has:

  - No lens distortion, no perspective shift, no focus/exposure
    drift: the (x, y) in the captured frame corresponds 1:1 to the
    host's pixel coordinates.
  - No "SMPTE bars" failure mode (the macOS "camera unavailable"
    placeholder that bit us multiple times during webcam debugging).
  - Per-frame latency dominated by the OS screenshot syscall: macOS
    ``screencapture -x`` is ~70-150 ms, Pillow ``ImageGrab`` on
    X11/Wayland is ~30-100 ms.
  - A trivial cursor problem: the cursor IS in the captured frame
    at its exact pixel position. The existing oscillation-variance
    detector still works, but a much simpler template match (or
    even pre-known cursor coordinates queried from the OS) would
    work too.

Implementation: prefers Pillow's ``ImageGrab.grab()`` because it's
already a project dep and works on macOS + Windows + X11/Linux
without shelling out. Falls back to the ``screencapture`` CLI on
macOS if Pillow's grab fails. No native libraries / pyobjc / Quartz
needed.

To use, set ``capture_source = "screen"`` in the target's
``targets.toml`` entry. ``camera_index`` is then ignored.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handsneyes.core.capture.base import (
    CaptureError,
    CapturedFrame,
    CaptureSource,
    CropRegion,
)

logger = logging.getLogger(__name__)


def _looks_like_test_pattern(img: np.ndarray) -> bool:
    """Detect macOS' "no Screen Recording permission" placeholder.

    The placeholder is SMPTE-style vertical color bars; every column
    is a constant intensity, so the per-pixel vertical gradient is
    exactly 0. Real desktop captures average ~10 even on minimal
    desktops (icons, menu bar, window edges). Threshold of 1.5 has
    ample margin.
    """
    if img is None or img.size == 0:
        return False
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vert = float(np.abs(np.diff(gray.astype(np.int16), axis=0)).mean())
    return vert < 1.5


class ScreenCapture(CaptureSource):
    """Captures frames of the local machine's display.

    The capture is done out-of-process (Pillow ImageGrab → numpy →
    cv2 colour layout) so there's no display-server dep at import
    time and no event loop coupling. Each ``capture_frame`` is one
    OS screenshot syscall.

    Args:
        display_index: 0 for the primary display. On macOS multi-
            monitor setups, Pillow's ImageGrab returns the union
            bounding box; pass a non-zero ``display_index`` to use
            ``screencapture -D <n+1>`` (1-based) which can capture
            a single display.
        crop_region: passed through to the base class. Useful when
            the operator only cares about a specific region (a
            specific app window etc.).
        resolution: ignored; ScreenCapture always returns at the
            display's native resolution.
    """

    def __init__(
        self,
        display_index: int = 0,
        crop_region: CropRegion | None = None,
        resolution: tuple[int, int] | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__(crop_region=crop_region)
        self._display_index = display_index
        self._actual_w: int = 0
        self._actual_h: int = 0
        self._sysname = platform.system()
        # Detect Pillow availability once at init so capture_frame
        # doesn't repeatedly try-import in the hot path.
        try:
            from PIL import ImageGrab  # noqa: F401
            self._have_pil = True
        except Exception:
            self._have_pil = False

    async def open(self) -> None:
        """Probe-capture one frame to seed dimensions + confirm we can
        actually grab. Same lifecycle contract as WebcamCapture."""
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self._grab_sync)
        if frame is None or frame.size == 0:
            raise CaptureError(
                "ScreenCapture: probe grab returned empty frame "
                f"(display_index={self._display_index})",
            )
        if _looks_like_test_pattern(frame):
            raise CaptureError(
                "ScreenCapture got the macOS Screen Recording "
                "placeholder (SMPTE-style color bars). Open "
                "System Settings → Privacy & Security → Screen "
                "Recording and add the Python interpreter / terminal "
                "that launched handsneyes, then restart the cc.",
            )
        self._actual_h, self._actual_w = frame.shape[:2]
        self._is_open = True
        logger.info(
            "Opened screen capture (display=%d, %dx%d, backend=%s)",
            self._display_index,
            self._actual_w, self._actual_h,
            "pillow" if self._have_pil else "screencapture-cli",
        )

    async def close(self) -> None:
        """No-op: there's no resource to release between captures.
        Implemented for the CaptureSource contract."""
        self._is_open = False

    async def capture_frame(self) -> CapturedFrame:
        if not self._is_open:
            raise CaptureError("ScreenCapture is not open")
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self._grab_sync)
        if self._crop_region is not None:
            frame = self._apply_crop(frame)
        self._frame_counter += 1
        return CapturedFrame(
            image=frame,
            timestamp=datetime.now(),
            frame_number=self._frame_counter,
            source_device=f"screen:{self._display_index}",
            crop_applied=self._crop_region,
        )

    # ── backends ──────────────────────────────────────────────────

    def _grab_sync(self) -> np.ndarray:
        """Synchronous screenshot, returns a BGR ndarray.

        Order: Pillow ImageGrab (preferred — pure-Python, no shell) →
        macOS screencapture CLI (fallback if Pillow ImageGrab fails;
        macOS only). Linux/X11 falls back to ``import`` from
        ImageMagick if Pillow isn't installed."""
        if self._have_pil:
            try:
                return self._grab_pillow()
            except Exception as e:
                logger.debug("Pillow grab failed, falling back: %s", e)
        if self._sysname == "Darwin":
            return self._grab_screencapture_cli()
        raise CaptureError(
            f"ScreenCapture: no working backend on {self._sysname}. "
            "Install Pillow (already a project dep): pip install Pillow",
        )

    def _grab_pillow(self) -> np.ndarray:
        from PIL import ImageGrab
        # macOS multi-display: PIL grabs all_screens=True only on
        # Windows (the rare doc'd quirk). On macOS we get the
        # primary display anyway. display_index > 0 is currently
        # accepted but routes through screencapture if PIL gives us
        # the wrong thing — that's the operator's escape hatch.
        if self._sysname == "Darwin" and self._display_index != 0:
            return self._grab_screencapture_cli()
        img = ImageGrab.grab(all_screens=False)
        # PIL is RGB; cv2 is BGR.
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def _grab_screencapture_cli(self) -> np.ndarray:
        """macOS ``screencapture`` shell fallback. Writes to a tmp
        PNG and reads it back. Slower than Pillow but works even when
        Pillow's ImageGrab is broken (older macOS, unsigned binary
        without screen-recording permission, etc.)."""
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False,
        ) as f:
            tmp = f.name
        try:
            cmd = ["screencapture", "-x", "-t", "png"]
            # 1-based display index on screencapture's -D flag.
            if self._display_index > 0:
                cmd.extend(["-D", str(self._display_index + 1)])
            cmd.append(tmp)
            r = subprocess.run(
                cmd, check=False, capture_output=True, timeout=5.0,
            )
            if r.returncode != 0:
                stderr = (r.stderr or b"").decode("utf-8", errors="replace")
                # macOS's standard "no Screen Recording permission"
                # signature. Surface a useful pointer instead of the
                # raw stderr from a CLI most users don't recognise.
                if "could not create image from display" in stderr:
                    raise CaptureError(
                        "macOS Screen Recording permission not granted "
                        "to this Python process. Open System Settings → "
                        "Privacy & Security → Screen Recording and add "
                        "the terminal / IDE / Python interpreter you "
                        "launched handsneyes with. Then restart the cc.",
                    )
                raise CaptureError(
                    "screencapture failed: rc="
                    f"{r.returncode} stderr={stderr[:200]!r}",
                )
            img = cv2.imread(tmp, cv2.IMREAD_COLOR)
            if img is None:
                raise CaptureError(
                    f"screencapture wrote unreadable image at {tmp}",
                )
            return img
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        r = self._crop_region
        assert r is not None
        return frame[r.y : r.y + r.height, r.x : r.x + r.width].copy()
