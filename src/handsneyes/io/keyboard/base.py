"""Abstract base class for keyboard action output.

All keyboard output backends must conform to this interface, enabling
the system to swap between the HTTP backend (for the local endpoint)
and the future USB HID backend (for Raspberry Pi) without changing
any other code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class KeyboardOutput(ABC):
    """Abstract interface for sending keyboard actions to a target.

    Implementations translate logical keyboard actions (keystrokes,
    key combinations, text input) into the appropriate protocol for
    their target (HTTP requests, USB HID reports, etc.).

    Example usage::

        async with HttpKeyboardOutput(base_url="http://localhost:8080") as kb:
            await kb.send_text("ls -la")
            await kb.send_keystroke("Enter")
            await kb.send_key_combo(["ctrl"], "c")
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the keyboard output target.

        Raises:
            KeyboardOutputError: If connection cannot be established.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection to the output target."""
        ...

    @abstractmethod
    async def send_keystroke(self, key: str) -> None:
        """Send a single key press.

        Args:
            key: The key to press. Standard names include printable
                 characters, special keys ('Enter', 'Tab', 'Escape',
                 'Backspace', arrows, 'Home', 'End', PageUp/Down),
                 and function keys ('F1'..'F12').

        Raises:
            KeyboardOutputError: If the keystroke cannot be sent.
        """
        ...

    @abstractmethod
    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination (modifier keys + main key).

        Args:
            modifiers: List of modifier keys to hold. Valid modifiers:
                       'ctrl', 'alt', 'shift', 'meta'/'super'/'win'.
            key: The main key to press while modifiers are held.

        Raises:
            KeyboardOutputError: If the combo cannot be sent.
        """
        ...

    @abstractmethod
    async def send_text(self, text: str, **kwargs: object) -> None:
        """Type a string of text character by character.

        Args:
            text: The text string to type. Does NOT automatically press
                  Enter at the end.
            **kwargs: Backend-specific options. Standard names that
                concrete backends should accept:
                ``secret=True`` to redact the text from local logs,
                ``warmup=False`` to skip the per-character HID warmup.

        Raises:
            KeyboardOutputError: If text input fails.
        """
        ...

    async def send_line(self, text: str) -> None:
        """Type a string of text and press Enter."""
        await self.send_text(text)
        await self.send_keystroke("Enter")

    async def __aenter__(self) -> KeyboardOutput:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        await self.disconnect()


class KeyboardOutputError(Exception):
    """Raised when keyboard output fails."""

    def __init__(self, message: str, backend: str = "") -> None:
        super().__init__(message)
        self.backend = backend
