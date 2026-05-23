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

    async def click(self, button: str = "left") -> None:
        """Send a mouse button click."""
        await self._post(f"{self._prefix}/click", {"button": button})
        logger.debug("Mouse click: %s", button)

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
