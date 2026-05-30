"""HTTP mouse output backend.

Sends mouse actions as HTTP requests to the Pi REST API.
Supports both USB HID and Bluetooth HID transports.
"""

from __future__ import annotations

import logging

import httpx

from handsneyes.io.mouse.base import MouseOutput, MouseOutputError

logger = logging.getLogger(__name__)


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:
        pass
    text = response.text.strip()
    return text or response.reason_phrase or "no response body"


class HttpMouseOutput(MouseOutput):
    """Sends mouse actions to the Pi HTTP REST API.

    Args:
        base_url: Pi REST API base URL (e.g. "http://10.0.0.2:8080").
        timeout: HTTP request timeout in seconds.
        transport: "bt" for Bluetooth HID endpoints, "usb" for USB HID.
    """

    def __init__(
        self,
        base_url: str = "http://10.0.0.2:8080",
        timeout: float = 10.0,
        transport: str = "bt",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

        if transport == "bt":
            self._prefix = "/bt/mouse"
        else:
            self._prefix = "/mouse"

    async def connect(self) -> None:
        """Create the HTTP client and verify Pi connectivity."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            if self._transport == "bt":
                health = resp.json()
                if not health.get("bt_hid_connected", False):
                    raise MouseOutputError(
                        f"Pi at {self._base_url} reports bt_hid_connected=false — "
                        "no Bluetooth client paired to the Pi. Re-pair from the "
                        "target device, or restart bluetoothd on the Pi.",
                        backend="http",
                    )
            logger.info(
                "Mouse connected to %s (transport=%s)",
                self._base_url,
                self._transport,
            )
        except MouseOutputError:
            await self._client.aclose()
            self._client = None
            raise
        except Exception as e:
            await self._client.aclose()
            self._client = None
            raise MouseOutputError(
                f"Failed to connect to Pi: {e}", backend="http"
            ) from e

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("Mouse disconnected")

    async def move(self, dx: int, dy: int) -> None:
        """Send a relative mouse movement."""
        await self._post(f"{self._prefix}/move", {"x": dx, "y": dy})

    async def move_large(self, dx: int, dy: int) -> None:
        """Send a large relative mouse movement in a single POST.

        Equivalent to many ``move()`` calls split into ±127 chunks
        but performed Pi-side without per-chunk HTTP overhead. macOS
        sees a single high-velocity burst, applies its high-speed
        pointer-accel curve, and the cursor covers significantly
        more screen per HID unit than the chunked path that goes
        through MOVE_STEP_SIZE-throttled ``move()`` calls.

        Used by cruise mode in the visual servo homer. Falls back
        gracefully on backends that don't implement it (the base
        class supplies a chunked default).
        """
        await self._post(
            f"{self._prefix}/move_large", {"x": dx, "y": dy},
        )

    async def click(self, button: str = "left", count: int = 1) -> None:
        """Send a mouse button click. count > 1 fires a multi-click
        natively on the Pi with tight inter-click timing — better
        than dispatching N single clicks from the dev side, which
        adds HTTP roundtrip per click and risks pushing the press-
        to-press gap past macOS's double-click threshold.
        """
        await self._post(
            f"{self._prefix}/click",
            {"button": button, "count": count},
        )
        logger.debug("Mouse click: %s x%d", button, count)

    async def press(self, button: str = "left") -> None:
        """Hold a button down — paired with release() for drag flows."""
        await self._post(f"{self._prefix}/press", {"button": button})
        logger.debug("Mouse press: %s", button)

    async def release(self, button: str = "left") -> None:
        """Release a previously-pressed button."""
        await self._post(f"{self._prefix}/release", {"button": button})
        logger.debug("Mouse release: %s", button)

    async def scroll(self, amount: int) -> None:
        """Send a scroll wheel action."""
        await self._post(f"{self._prefix}/scroll", {"amount": amount})
        logger.debug("Mouse scroll: %d", amount)

    async def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        """Send a POST request to the Pi."""
        if self._client is None:
            raise MouseOutputError("Not connected to Pi", backend="http")
        try:
            resp = await self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            detail = _extract_detail(e.response)
            raise MouseOutputError(
                f"HTTP {e.response.status_code} from {path}: {detail}",
                backend="http",
            ) from e
        except httpx.HTTPError as e:
            raise MouseOutputError(
                f"HTTP request to {path} failed: {e}", backend="http"
            ) from e
