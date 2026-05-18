"""Keyboard I/O: abstract base + backends."""

from handsneyes.io.keyboard.backends.http import HttpKeyboardOutput
from handsneyes.io.keyboard.base import KeyboardOutput, KeyboardOutputError

__all__ = ["HttpKeyboardOutput", "KeyboardOutput", "KeyboardOutputError"]
