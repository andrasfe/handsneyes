"""Tests for handsneyes.io.keyboard base + the afferent-backed HTTP backend.

The backend is now a thin async adapter over afferent.GatewayClient — these
tests mock the client and assert delegation, host routing, secret redaction,
and lifecycle/error behaviour.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from handsneyes.io.keyboard import (
    HttpKeyboardOutput,
    KeyboardOutput,
    KeyboardOutputError,
)


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        KeyboardOutput()  # type: ignore[abstract]


def _kb(**kw) -> "tuple[HttpKeyboardOutput, MagicMock]":
    k = HttpKeyboardOutput(**kw)
    gw = MagicMock()
    k._gw = gw
    return k, gw


class TestInit:
    def test_defaults(self) -> None:
        kb = HttpKeyboardOutput()
        assert kb._gw.base_url == "http://localhost:8080"

    def test_usb_transport_rejected(self) -> None:
        with pytest.raises(KeyboardOutputError):
            HttpKeyboardOutput(transport="usb")

    def test_custom_base_url_trailing_slash_stripped(self) -> None:
        kb = HttpKeyboardOutput(base_url="http://test/")
        assert kb._gw.base_url == "http://test"

    def test_host_mac_routes(self) -> None:
        kb = HttpKeyboardOutput(host_mac="aa:bb:cc:dd:ee:ff")
        assert kb._gw.host_mac == "AA:BB:CC:DD:EE:FF"


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        kb, gw = _kb()
        gw.is_hid_up.return_value = True
        await kb.connect()
        gw.is_hid_up.assert_called_once()
        assert kb._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure_when_hid_down(self) -> None:
        kb, gw = _kb()
        gw.is_hid_up.return_value = False
        with pytest.raises(KeyboardOutputError):
            await kb.connect()


class TestSend:
    @pytest.mark.asyncio
    async def test_keystroke_delegates(self) -> None:
        kb, gw = _kb()
        await kb.send_keystroke("Enter")
        gw.keystroke.assert_called_once_with("Enter")

    @pytest.mark.asyncio
    async def test_key_combo_delegates(self) -> None:
        kb, gw = _kb()
        await kb.send_key_combo(["meta"], "c")
        gw.key_combo.assert_called_once_with(["meta"], "c")

    @pytest.mark.asyncio
    async def test_send_text_delegates_with_warmup(self) -> None:
        kb, gw = _kb()
        await kb.send_text("hello")
        gw.text.assert_called_once_with("hello", warmup=True)

    @pytest.mark.asyncio
    async def test_send_text_warmup_false(self) -> None:
        kb, gw = _kb()
        await kb.send_text("https://x", warmup=False)
        gw.text.assert_called_once_with("https://x", warmup=False)

    @pytest.mark.asyncio
    async def test_send_text_secret_redacts_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kb, _ = _kb()
        caplog.set_level(logging.DEBUG, logger="handsneyes.io.keyboard")
        secret = "hunter2-very-private"
        await kb.send_text(secret, secret=True)
        blob = "\n".join(rec.getMessage() for rec in caplog.records)
        assert secret not in blob
        assert "redacted" in blob


class TestErrors:
    @pytest.mark.asyncio
    async def test_backend_unavailable_becomes_keyboard_error(self) -> None:
        from afferent import BackendUnavailable

        kb, gw = _kb()
        gw.keystroke.side_effect = BackendUnavailable("gateway unreachable")
        with pytest.raises(KeyboardOutputError):
            await kb.send_keystroke("a")


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        kb, _ = _kb()
        kb._connected = True
        await kb.disconnect()
        assert kb._connected is False
