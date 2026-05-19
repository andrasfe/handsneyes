"""Keyboard I/O: abstract base + backends + platform-aware proxy."""

from handsneyes.io.keyboard.backends.http import HttpKeyboardOutput
from handsneyes.io.keyboard.base import KeyboardOutput, KeyboardOutputError
from handsneyes.io.keyboard.platform_keyboard import PlatformKeyboard

__all__ = [
    "HttpKeyboardOutput",
    "KeyboardOutput",
    "KeyboardOutputError",
    "PlatformKeyboard",
]
