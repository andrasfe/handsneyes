"""Tests for handsneyes.io.mouse base + HTTP backend.

Ported from terminaleyes/tests/unit/test_mouse/test_http_backend.py
plus the abstract-base-class instantiation check required by ha-006.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from handsneyes.io.mouse import HttpMouseOutput, MouseOutput
from handsneyes.io.mouse.base import MouseOutputError


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        MouseOutput()  # type: ignore[abstract]


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock(spec=httpx.AsyncClient)


class TestInit:
    def test_defaults(self) -> None:
        mouse = HttpMouseOutput()
        assert mouse._base_url == "http://10.0.0.2:8080"
        assert mouse._transport == "bt"
        assert mouse._prefix == "/bt/mouse"

    def test_usb_transport(self) -> None:
        mouse = HttpMouseOutput(transport="usb")
        assert mouse._prefix == "/mouse"

    def test_custom_base_url(self) -> None:
        mouse = HttpMouseOutput(base_url="http://192.168.1.100:9090/")
        assert mouse._base_url == "http://192.168.1.100:9090"


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        mouse = HttpMouseOutput()
        with patch("handsneyes.io.mouse.backends.http.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            await mouse.connect()
            mock_instance.get.assert_called_once_with("/health")
            assert mouse._client is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self) -> None:
        mouse = HttpMouseOutput()
        with patch("handsneyes.io.mouse.backends.http.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value = mock_instance

            with pytest.raises(MouseOutputError, match="Failed to connect"):
                await mouse.connect()
            assert mouse._client is None


class TestMove:
    @pytest.mark.asyncio
    async def test_http_backend_move_via_post(self, mock_client: AsyncMock) -> None:
        mouse = HttpMouseOutput()
        mouse._client = mock_client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        await mouse.move(10, -5)
        mock_client.post.assert_called_once_with(
            "/bt/mouse/move", json={"x": 10, "y": -5}
        )

    @pytest.mark.asyncio
    async def test_move_usb_transport(self, mock_client: AsyncMock) -> None:
        mouse = HttpMouseOutput(transport="usb")
        mouse._client = mock_client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        await mouse.move(50, 50)
        mock_client.post.assert_called_once_with(
            "/mouse/move", json={"x": 50, "y": 50}
        )


class TestClick:
    @pytest.mark.asyncio
    async def test_http_backend_click_left_button(self, mock_client: AsyncMock) -> None:
        mouse = HttpMouseOutput()
        mouse._client = mock_client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        await mouse.click("left")
        mock_client.post.assert_called_once_with(
            "/bt/mouse/click", json={"button": "left"}
        )

    @pytest.mark.asyncio
    async def test_click_right(self, mock_client: AsyncMock) -> None:
        mouse = HttpMouseOutput()
        mouse._client = mock_client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        await mouse.click("right")
        mock_client.post.assert_called_once_with(
            "/bt/mouse/click", json={"button": "right"}
        )


class TestScroll:
    @pytest.mark.asyncio
    async def test_scroll(self, mock_client: AsyncMock) -> None:
        mouse = HttpMouseOutput()
        mouse._client = mock_client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        await mouse.scroll(-3)
        mock_client.post.assert_called_once_with(
            "/bt/mouse/scroll", json={"amount": -3}
        )


class TestNotConnected:
    @pytest.mark.asyncio
    async def test_move_not_connected(self) -> None:
        mouse = HttpMouseOutput()
        with pytest.raises(MouseOutputError, match="Not connected"):
            await mouse.move(10, 10)

    @pytest.mark.asyncio
    async def test_click_not_connected(self) -> None:
        mouse = HttpMouseOutput()
        with pytest.raises(MouseOutputError, match="Not connected"):
            await mouse.click()


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_client: AsyncMock) -> None:
        mouse = HttpMouseOutput()
        mouse._client = mock_client

        await mouse.disconnect()
        mock_client.aclose.assert_called_once()
        assert mouse._client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        mouse = HttpMouseOutput()
        await mouse.disconnect()
