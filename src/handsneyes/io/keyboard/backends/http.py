"""Keyboard output over the afferent BT HID gateway.

Thin async adapter over ``afferent.GatewayClient`` — the HTTP transport,
per-request host routing, and error handling all live in afferent now (the
gateway itself is ``afferent.gateway``). handsneyes already speaks gateway-
native key names (the platform adapter + ``PlatformKeyboard`` remap modifiers
and capitalise special keys before they reach here), so this layer just
forwards. The sync client is run in a worker thread so HID calls don't block
the event loop.
"""

from __future__ import annotations

import asyncio
import logging

from afferent import BackendUnavailable, GatewayClient

from handsneyes.io.keyboard.base import KeyboardOutput, KeyboardOutputError

logger = logging.getLogger(__name__)


class HttpKeyboardOutput(KeyboardOutput):
    """Sends keyboard actions to the Pi via the afferent gateway client.

    Args:
        base_url: gateway base URL (e.g. "http://10.0.0.2:8080").
        timeout: per-request timeout in seconds.
        transport: only "bt" is supported (the afferent gateway client speaks
            the Bluetooth-HID endpoints); kept for call-site compatibility.
        host_mac: target's Bluetooth MAC for multi-host routing, or None for
            the gateway's active/single-host default.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        transport: str = "bt",
        host_mac: str | None = None,
    ) -> None:
        if transport != "bt":
            raise KeyboardOutputError(
                f"transport {transport!r} not supported — the afferent "
                "gateway client drives Bluetooth HID (use transport='bt')",
                backend="afferent",
            )
        self._gw = GatewayClient(base_url, host_mac=host_mac, timeout=timeout)
        self._connected = False

    async def connect(self) -> None:
        """Verify the gateway is reachable and an HID link is open."""
        try:
            up = await asyncio.to_thread(self._gw.is_hid_up)
        except Exception as e:
            raise KeyboardOutputError(
                f"Failed to reach gateway at {self._gw.base_url}: {e}",
                backend="afferent",
            ) from e
        if not up:
            raise KeyboardOutputError(
                f"Gateway at {self._gw.base_url} reports no open HID link for "
                "this host — connect the target's Bluetooth (devmouse), or "
                "check the gateway.",
                backend="afferent",
            )
        self._connected = True
        logger.info("Keyboard connected to gateway %s", self._gw.base_url)

    async def disconnect(self) -> None:
        self._connected = False
        logger.debug("Keyboard disconnected")

    async def send_keystroke(self, key: str) -> None:
        await self._call(self._gw.keystroke, key)
        logger.debug("Sent keystroke: %s", key)

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        await self._call(self._gw.key_combo, modifiers, key)
        logger.debug("Sent key combo: %s+%s", "+".join(modifiers), key)

    async def send_text(self, text: str, **kwargs: object) -> None:
        secret = bool(kwargs.get("secret", False))
        warmup = bool(kwargs.get("warmup", True))
        await self._call(lambda: self._gw.text(text, warmup=warmup))
        if secret:
            logger.debug("Sent text (length=%d, redacted)", len(text))
        else:
            logger.debug("Sent text: %s", text[:50])

    async def _call(self, fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except BackendUnavailable as e:
            raise KeyboardOutputError(str(e), backend="afferent") from e
