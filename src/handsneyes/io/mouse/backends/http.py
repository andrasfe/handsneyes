"""Mouse output over the afferent BT HID gateway.

Thin async adapter over ``afferent.GatewayClient`` — the HTTP transport,
per-request host routing, and error handling live in afferent now. The sync
client is run in a worker thread so HID calls don't block the event loop.
``move_large`` maps to the gateway's single-burst endpoint (the visual servo's
cruise phase relies on it).
"""

from __future__ import annotations

import asyncio
import logging

from afferent import BackendUnavailable, GatewayClient

from handsneyes.io.mouse.base import MouseOutput, MouseOutputError

logger = logging.getLogger(__name__)


class HttpMouseOutput(MouseOutput):
    """Sends mouse actions to the Pi via the afferent gateway client.

    Args:
        base_url: gateway base URL (e.g. "http://10.0.0.2:8080").
        timeout: per-request timeout in seconds.
        transport: only "bt" is supported; kept for call-site compatibility.
        host_mac: target's Bluetooth MAC for multi-host routing, or None.
    """

    def __init__(
        self,
        base_url: str = "http://10.0.0.2:8080",
        timeout: float = 10.0,
        transport: str = "bt",
        host_mac: str | None = None,
    ) -> None:
        if transport != "bt":
            raise MouseOutputError(
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
            raise MouseOutputError(
                f"Failed to reach gateway at {self._gw.base_url}: {e}",
                backend="afferent",
            ) from e
        if not up:
            raise MouseOutputError(
                f"Gateway at {self._gw.base_url} reports no open HID link for "
                "this host — connect the target's Bluetooth (devmouse), or "
                "check the gateway.",
                backend="afferent",
            )
        self._connected = True
        logger.info("Mouse connected to gateway %s", self._gw.base_url)

    async def disconnect(self) -> None:
        self._connected = False
        logger.debug("Mouse disconnected")

    async def move(self, dx: int, dy: int) -> None:
        await self._call(self._gw.move, dx, dy)

    async def move_large(self, dx: int, dy: int) -> None:
        """One high-velocity burst, chunked Pi-side in a single POST — the
        homer's cruise phase depends on the fast-pointer-accel curve macOS
        applies to it (far more screen per HID unit than throttled moves)."""
        await self._call(self._gw.move_large, dx, dy)

    async def click(self, button: str = "left", count: int = 1) -> None:
        await self._call(self._gw.click, button, count)
        logger.debug("Mouse click: %s x%d", button, count)

    async def press(self, button: str = "left") -> None:
        await self._call(self._gw.press, button)
        logger.debug("Mouse press: %s", button)

    async def release(self, button: str = "left") -> None:
        await self._call(self._gw.release, button)
        logger.debug("Mouse release: %s", button)

    async def scroll(self, amount: int) -> None:
        await self._call(self._gw.scroll, amount)
        logger.debug("Mouse scroll: %d", amount)

    async def _call(self, fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except BackendUnavailable as e:
            raise MouseOutputError(str(e), backend="afferent") from e
