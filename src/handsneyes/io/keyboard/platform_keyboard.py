"""PlatformKeyboard: keyboard proxy that applies adapter shortcut remap.

Wraps any :class:`KeyboardOutput`. ``send_key_combo`` calls
:meth:`PlatformAdapter.remap_combo` before delegating, so a logical
chord like ``ctrl+a`` becomes the OS-native chord (``cmd+a`` on macOS,
identity on Linux/GNOME) without every agent re-implementing the
mapping.

A ``send_raw_combo`` escape hatch bypasses the remap for the rare case
that wants the literal chord (regression tests, OS-specific tooling).
Everything else (``send_keystroke``, ``send_text``, ``send_line``,
``connect``, ``disconnect``, context-manager protocol) is a pass-through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from handsneyes.io.keyboard.base import KeyboardOutput

if TYPE_CHECKING:
    from handsneyes.platforms.base import PlatformAdapter


class PlatformKeyboard(KeyboardOutput):
    """Adapter-aware proxy around a concrete keyboard backend."""

    def __init__(
        self,
        inner: KeyboardOutput,
        adapter: PlatformAdapter,
    ) -> None:
        self._inner = inner
        self._adapter = adapter

    async def connect(self) -> None:
        await self._inner.connect()

    async def disconnect(self) -> None:
        await self._inner.disconnect()

    async def send_keystroke(self, key: str) -> None:
        await self._inner.send_keystroke(key)

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        remapped_mods, remapped_key = self._adapter.remap_combo(
            list(modifiers), key
        )
        await self._inner.send_key_combo(remapped_mods, remapped_key)

    async def send_raw_combo(self, modifiers: list[str], key: str) -> None:
        """Send the literal chord without consulting the adapter."""
        await self._inner.send_key_combo(modifiers, key)

    async def send_text(self, text: str, **kwargs: object) -> None:
        # Forward arbitrary keyword args (secret=, warmup=) — backends
        # accept different sets and we don't want to second-guess them.
        await self._inner.send_text(text, **kwargs)
