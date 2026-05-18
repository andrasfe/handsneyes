"""Tests for handsneyes.io.keyboard base + HTTP backend."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from handsneyes.io.keyboard import (
    HttpKeyboardOutput,
    KeyboardOutput,
    KeyboardOutputError,
)


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        KeyboardOutput()  # type: ignore[abstract]


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock(spec=httpx.AsyncClient)


class TestInit:
    def test_defaults(self) -> None:
        kb = HttpKeyboardOutput()
        assert kb._base_url == "http://localhost:8080"
        assert kb._transport == "usb"
        assert kb._prefix == ""

    def test_bt_transport(self) -> None:
        kb = HttpKeyboardOutput(transport="bt")
        assert kb._prefix == "/bt"

    def test_custom_base_url_trailing_slash_stripped(self) -> None:
        kb = HttpKeyboardOutput(base_url="http://test/")
        assert kb._base_url == "http://test"


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        kb = HttpKeyboardOutput()
        with patch(
            "handsneyes.io.keyboard.backends.http.httpx.AsyncClient"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            await kb.connect()
            mock_instance.get.assert_called_once_with("/health")
            assert kb._client is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self) -> None:
        kb = HttpKeyboardOutput()
        with patch(
            "handsneyes.io.keyboard.backends.http.httpx.AsyncClient"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                side_effect=httpx.ConnectError("refused")
            )
            mock_cls.return_value = mock_instance

            with pytest.raises(KeyboardOutputError, match="Failed to connect"):
                await kb.connect()
            assert kb._client is None


class TestSend:
    @pytest.mark.asyncio
    async def test_keystroke_via_post(
        self, mock_client: AsyncMock
    ) -> None:
        kb = HttpKeyboardOutput()
        kb._client = mock_client
        mock_client.post = AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock()
        ))
        await kb.send_keystroke("Enter")
        mock_client.post.assert_called_once_with(
            "/keystroke", json={"key": "Enter"}
        )

    @pytest.mark.asyncio
    async def test_key_combo_via_post(
        self, mock_client: AsyncMock
    ) -> None:
        kb = HttpKeyboardOutput()
        kb._client = mock_client
        mock_client.post = AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock()
        ))
        await kb.send_key_combo(["ctrl"], "c")
        mock_client.post.assert_called_once_with(
            "/key-combo", json={"modifiers": ["ctrl"], "key": "c"}
        )

    @pytest.mark.asyncio
    async def test_send_text_via_post(
        self, mock_client: AsyncMock
    ) -> None:
        kb = HttpKeyboardOutput()
        kb._client = mock_client
        mock_client.post = AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock()
        ))
        await kb.send_text("hello")
        mock_client.post.assert_called_once_with(
            "/text", json={"text": "hello", "warmup": True}
        )

    @pytest.mark.asyncio
    async def test_send_text_secret_redacts_logs(
        self,
        mock_client: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        kb = HttpKeyboardOutput()
        kb._client = mock_client
        mock_client.post = AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock()
        ))
        caplog.set_level(logging.DEBUG, logger="handsneyes.io.keyboard")
        secret = "hunter2-very-private"
        await kb.send_text(secret, secret=True)
        text_blob = "\n".join(rec.getMessage() for rec in caplog.records)
        assert secret not in text_blob
        assert "redacted" in text_blob


class TestNotConnected:
    @pytest.mark.asyncio
    async def test_keystroke_not_connected(self) -> None:
        kb = HttpKeyboardOutput()
        with pytest.raises(KeyboardOutputError, match="Not connected"):
            await kb.send_keystroke("a")


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_client: AsyncMock) -> None:
        kb = HttpKeyboardOutput()
        kb._client = mock_client
        await kb.disconnect()
        mock_client.aclose.assert_called_once()
        assert kb._client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        kb = HttpKeyboardOutput()
        await kb.disconnect()
