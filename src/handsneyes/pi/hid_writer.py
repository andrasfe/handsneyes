# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/raspi; lint cleanup deferred.
"""Low-level USB HID report writers for /dev/hidg0 (keyboard) and /dev/hidg1 (mouse).

Keyboard (/dev/hidg0): 8-byte reports
    [modifier, 0x00, key1, key2, key3, key4, key5, key6]

Mouse (/dev/hidg1): 4-byte reports
    [buttons, x_delta, y_delta, wheel]

A key press is sent by writing a report with the key, then a release
is sent by writing an all-zeros report.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from enum import IntFlag
from pathlib import Path

from handsneyes.pi.hid_codes import (
    MODIFIER_LEFT_SHIFT,
    MODIFIER_NONE,
    SHIFT_CHARS,
    char_to_hid,
    key_name_to_hid,
    modifiers_to_bitmask,
)

logger = logging.getLogger(__name__)

# 8-byte empty report = all keys released
RELEASE_REPORT = b"\x00" * 8

# Default delay between key press and release (seconds)
DEFAULT_KEYPRESS_DELAY = 0.02
# Default delay between consecutive characters when typing text
DEFAULT_INTER_CHAR_DELAY = 0.01


class HidWriteError(Exception):
    """Raised when writing to the HID device fails."""


class HidWriter:
    """Writes USB HID keyboard reports to /dev/hidg0.

    Usage::

        writer = HidWriter()
        await writer.open()
        await writer.press_key(0x00, 0x28)  # Enter
        await writer.release_keys()
        await writer.close()
    """

    def __init__(
        self,
        device_path: str = "/dev/hidg0",
        keypress_delay: float = DEFAULT_KEYPRESS_DELAY,
        inter_char_delay: float = DEFAULT_INTER_CHAR_DELAY,
    ) -> None:
        self._device_path = Path(device_path)
        self._keypress_delay = keypress_delay
        self._inter_char_delay = inter_char_delay
        self._fd: int | None = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    async def open(self) -> None:
        """Open the HID gadget device for writing."""
        import os

        try:
            # Open in non-blocking write mode; run in executor since it's I/O
            loop = asyncio.get_running_loop()
            self._fd = await loop.run_in_executor(
                None, lambda: os.open(str(self._device_path), os.O_WRONLY)
            )
            logger.info("Opened HID device: %s", self._device_path)
        except OSError as e:
            raise HidWriteError(
                f"Cannot open HID device {self._device_path}: {e}"
            ) from e

    async def close(self) -> None:
        """Release all keys and close the device."""
        import os

        if self._fd is not None:
            try:
                await self.release_keys()
            except HidWriteError:
                pass
            try:
                loop = asyncio.get_running_loop()
                fd = self._fd
                await loop.run_in_executor(None, lambda: os.close(fd))
            except OSError:
                pass
            self._fd = None
            logger.info("Closed HID device")

    async def _write_report(self, report: bytes) -> None:
        """Write an 8-byte HID report to the device."""
        import os

        if self._fd is None:
            raise HidWriteError("HID device not open")
        if len(report) != 8:
            raise HidWriteError(f"HID report must be 8 bytes, got {len(report)}")
        try:
            loop = asyncio.get_running_loop()
            fd = self._fd
            await loop.run_in_executor(None, lambda: os.write(fd, report))
        except OSError as e:
            raise HidWriteError(f"Failed to write HID report: {e}") from e

    async def press_key(self, modifier: int, scan_code: int) -> None:
        """Send a key-press report (modifier + one key)."""
        report = bytes([modifier, 0x00, scan_code, 0x00, 0x00, 0x00, 0x00, 0x00])
        await self._write_report(report)

    async def release_keys(self) -> None:
        """Send an all-zeros report (release all keys)."""
        await self._write_report(RELEASE_REPORT)

    async def tap_key(self, modifier: int, scan_code: int) -> None:
        """Press and release a key with appropriate timing."""
        await self.press_key(modifier, scan_code)
        await asyncio.sleep(self._keypress_delay)
        await self.release_keys()

    async def send_keystroke(self, key: str) -> None:
        """Send a named key (e.g., 'Enter', 'Tab', 'a')."""
        # Check if it's a shifted character
        if key in SHIFT_CHARS:
            modifier, scan_code = char_to_hid(key)
        elif len(key) == 1:
            modifier, scan_code = char_to_hid(key)
        else:
            scan_code = key_name_to_hid(key)
            modifier = MODIFIER_NONE
        await self.tap_key(modifier, scan_code)
        logger.debug("Sent keystroke: %s (mod=0x%02X scan=0x%02X)", key, modifier, scan_code)

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination (e.g., ctrl+c)."""
        mod_bitmask = modifiers_to_bitmask(modifiers)
        # For single chars, check if the key itself needs shift
        if key in SHIFT_CHARS:
            base_char = SHIFT_CHARS[key]
            scan_code = key_name_to_hid(base_char)
            mod_bitmask |= MODIFIER_LEFT_SHIFT
        else:
            scan_code = key_name_to_hid(key)
        await self.tap_key(mod_bitmask, scan_code)
        logger.debug(
            "Sent combo: %s+%s (mod=0x%02X scan=0x%02X)",
            "+".join(modifiers), key, mod_bitmask, scan_code,
        )

    async def send_text(self, text: str) -> None:
        """Type a string character by character."""
        for char in text:
            modifier, scan_code = char_to_hid(char)
            await self.tap_key(modifier, scan_code)
            await asyncio.sleep(self._inter_char_delay)
        logger.debug("Sent text: %s", text[:50])

    async def __aenter__(self) -> HidWriter:
        await self.open()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Mouse HID writer (/dev/hidg1)
# ---------------------------------------------------------------------------

class MouseButton(IntFlag):
    """Mouse button bitmask for the HID report."""
    NONE = 0x00
    LEFT = 0x01
    RIGHT = 0x02
    MIDDLE = 0x04


BUTTON_MAP: dict[str, MouseButton] = {
    "left": MouseButton.LEFT,
    "right": MouseButton.RIGHT,
    "middle": MouseButton.MIDDLE,
}

# 4-byte empty report = all buttons released, no movement
MOUSE_RELEASE_REPORT = b"\x00" * 4


def _clamp(value: int, minimum: int = -127, maximum: int = 127) -> int:
    return max(minimum, min(maximum, value))


class MouseHidWriter:
    """Writes USB HID mouse reports to /dev/hidg1.

    4-byte reports: [buttons, x_delta, y_delta, wheel]

    Usage::

        writer = MouseHidWriter()
        await writer.open()
        await writer.move(10, -5)
        await writer.click("left")
        await writer.scroll(-3)
        await writer.close()
    """

    def __init__(self, device_path: str = "/dev/hidg1") -> None:
        self._device_path = Path(device_path)
        self._fd: int | None = None
        self._buttons: int = 0

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    async def open(self) -> None:
        """Open the HID gadget device for writing."""
        import os

        try:
            loop = asyncio.get_running_loop()
            self._fd = await loop.run_in_executor(
                None, lambda: os.open(str(self._device_path), os.O_WRONLY)
            )
            logger.info("Opened mouse HID device: %s", self._device_path)
        except OSError as e:
            raise HidWriteError(
                f"Cannot open mouse HID device {self._device_path}: {e}"
            ) from e

    async def close(self) -> None:
        """Release all buttons and close the device."""
        import os

        if self._fd is not None:
            try:
                self._buttons = 0
                await self._write_report(MOUSE_RELEASE_REPORT)
            except HidWriteError:
                pass
            try:
                loop = asyncio.get_running_loop()
                fd = self._fd
                await loop.run_in_executor(None, lambda: os.close(fd))
            except OSError:
                pass
            self._fd = None
            logger.info("Closed mouse HID device")

    async def _write_report(self, report: bytes) -> None:
        """Write a 4-byte HID mouse report to the device."""
        import os

        if self._fd is None:
            raise HidWriteError("Mouse HID device not open")
        if len(report) != 4:
            raise HidWriteError(f"Mouse HID report must be 4 bytes, got {len(report)}")
        try:
            loop = asyncio.get_running_loop()
            fd = self._fd
            await loop.run_in_executor(None, lambda: os.write(fd, report))
        except OSError as e:
            raise HidWriteError(f"Failed to write mouse HID report: {e}") from e

    async def move(self, x: int, y: int) -> None:
        """Move the mouse cursor by (x, y) relative pixels."""
        x = _clamp(x)
        y = _clamp(y)
        report = struct.pack("BbbB", self._buttons, x, y, 0)
        await self._write_report(report)
        logger.debug("Mouse move: dx=%d dy=%d", x, y)

    async def click(self, button: str = "left") -> None:
        """Click a mouse button (press and release)."""
        btn = BUTTON_MAP.get(button.lower())
        if btn is None:
            raise ValueError(f"Unknown button: {button!r}. Use: left, right, middle")
        # Press
        self._buttons |= btn
        report = struct.pack("BbbB", self._buttons, 0, 0, 0)
        await self._write_report(report)
        await asyncio.sleep(0.05)
        # Release
        self._buttons &= ~btn
        report = struct.pack("BbbB", self._buttons, 0, 0, 0)
        await self._write_report(report)
        logger.debug("Mouse click: %s", button)

    async def scroll(self, amount: int) -> None:
        """Scroll the mouse wheel. Positive=up, negative=down."""
        amount = _clamp(amount)
        report = struct.pack("Bbb b", self._buttons, 0, 0, amount)
        await self._write_report(report)
        logger.debug("Mouse scroll: %d", amount)

    async def __aenter__(self) -> MouseHidWriter:
        await self.open()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        await self.close()
