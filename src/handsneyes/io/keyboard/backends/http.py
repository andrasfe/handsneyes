"""HTTP keyboard output backend.

Sends keyboard actions as HTTP requests to the local command endpoint.
"""

from __future__ import annotations

import logging

import httpx

from handsneyes.io.keyboard.base import KeyboardOutput, KeyboardOutputError

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


class HttpKeyboardOutput(KeyboardOutput):
    """Sends keyboard actions to the local HTTP command endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        transport: str = "usb",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._prefix = "/bt" if transport == "bt" else ""
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Create the HTTP client and verify endpoint connectivity."""
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
                    raise KeyboardOutputError(
                        f"Pi at {self._base_url} reports bt_hid_connected=false — "
                        "no Bluetooth client paired to the Pi. Re-pair from the "
                        "target device, or restart bluetoothd on the Pi.",
                        backend="http",
                    )
            logger.info("Connected to endpoint at %s", self._base_url)
        except KeyboardOutputError:
            await self._client.aclose()
            self._client = None
            raise
        except Exception as e:
            await self._client.aclose()
            self._client = None
            raise KeyboardOutputError(
                f"Failed to connect to endpoint: {e}", backend="http"
            ) from e

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("Disconnected from endpoint")

    async def send_keystroke(self, key: str) -> None:
        """Send a keystroke via HTTP POST."""
        await self._post(f"{self._prefix}/keystroke", {"key": key})
        logger.debug("Sent keystroke: %s", key)

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination via HTTP POST."""
        await self._post(
            f"{self._prefix}/key-combo", {"modifiers": modifiers, "key": key}
        )
        logger.debug("Sent key combo: %s+%s", "+".join(modifiers), key)

    async def send_text(  # type: ignore[override]
        self,
        text: str,
        *,
        secret: bool = False,
        warmup: bool = True,
    ) -> None:
        """Send text input via HTTP POST.

        ``secret=True`` redacts the text from any logs — use when typing
        passwords or other sensitive content. The Pi side already only
        logs the length.

        ``warmup=False`` tells the Pi to skip the double-tap-with-
        backspace warmup on the first character. Use this for input
        contexts (e.g. some browser URL bars) where Backspace is bound
        to back-navigation rather than character deletion, which would
        otherwise produce a doubled first character.
        """
        await self._post(
            f"{self._prefix}/text",
            {"text": text, "warmup": warmup},
        )
        if secret:
            logger.debug("Sent text (length=%d, redacted)", len(text))
        else:
            logger.debug("Sent text: %s", text[:50])

    async def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        """Send a POST request to the endpoint."""
        if self._client is None:
            raise KeyboardOutputError(
                "Not connected to endpoint", backend="http"
            )
        try:
            resp = await self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            detail = _extract_detail(e.response)
            raise KeyboardOutputError(
                f"HTTP {e.response.status_code} from {path}: {detail}",
                backend="http",
            ) from e
        except httpx.HTTPError as e:
            raise KeyboardOutputError(
                f"HTTP request to {path} failed: {e}", backend="http"
            ) from e
