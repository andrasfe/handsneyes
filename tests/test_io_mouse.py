"""Tests for handsneyes.io.mouse base + the afferent-backed HTTP backend.

The backend is now a thin async adapter over afferent.GatewayClient — these
tests mock the client and assert each method delegates with the right args,
host routing flows through, and lifecycle/error behaviour holds.
"""

from unittest.mock import MagicMock, patch

import pytest

from handsneyes.io.mouse import HttpMouseOutput, MouseOutput
from handsneyes.io.mouse.base import MouseOutputError


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        MouseOutput()  # type: ignore[abstract]


def _mouse(**kw) -> "tuple[HttpMouseOutput, MagicMock]":
    """A backend with its GatewayClient swapped for a MagicMock."""
    m = HttpMouseOutput(**kw)
    gw = MagicMock()
    m._gw = gw
    return m, gw


class TestInit:
    def test_defaults(self) -> None:
        mouse = HttpMouseOutput()
        assert mouse._gw.base_url == "http://10.0.0.2:8080"

    def test_usb_transport_rejected(self) -> None:
        with pytest.raises(MouseOutputError):
            HttpMouseOutput(transport="usb")

    def test_custom_base_url(self) -> None:
        mouse = HttpMouseOutput(base_url="http://192.168.1.100:9090/")
        assert mouse._gw.base_url == "http://192.168.1.100:9090"

    def test_host_mac_routes(self) -> None:
        mouse = HttpMouseOutput(host_mac="aa:bb:cc:dd:ee:ff")
        assert mouse._gw.host_mac == "AA:BB:CC:DD:EE:FF"


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        mouse, gw = _mouse()
        gw.is_hid_up.return_value = True
        await mouse.connect()
        gw.is_hid_up.assert_called_once()
        assert mouse._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure_when_hid_down(self) -> None:
        mouse, gw = _mouse()
        gw.is_hid_up.return_value = False
        with pytest.raises(MouseOutputError):
            await mouse.connect()


class TestActions:
    @pytest.mark.asyncio
    async def test_move(self) -> None:
        mouse, gw = _mouse()
        await mouse.move(10, -5)
        gw.move.assert_called_once_with(10, -5)

    @pytest.mark.asyncio
    async def test_move_large(self) -> None:
        mouse, gw = _mouse()
        await mouse.move_large(2000, -1500)
        gw.move_large.assert_called_once_with(2000, -1500)

    @pytest.mark.asyncio
    async def test_click_left(self) -> None:
        mouse, gw = _mouse()
        await mouse.click("left")
        gw.click.assert_called_once_with("left", 1)

    @pytest.mark.asyncio
    async def test_click_count(self) -> None:
        mouse, gw = _mouse()
        await mouse.click("right", 2)
        gw.click.assert_called_once_with("right", 2)

    @pytest.mark.asyncio
    async def test_press_release(self) -> None:
        mouse, gw = _mouse()
        await mouse.press("left")
        await mouse.release("left")
        gw.press.assert_called_once_with("left")
        gw.release.assert_called_once_with("left")

    @pytest.mark.asyncio
    async def test_scroll(self) -> None:
        mouse, gw = _mouse()
        await mouse.scroll(-3)
        gw.scroll.assert_called_once_with(-3)


class TestErrors:
    @pytest.mark.asyncio
    async def test_backend_unavailable_becomes_mouse_error(self) -> None:
        from afferent import BackendUnavailable

        mouse, gw = _mouse()
        gw.move.side_effect = BackendUnavailable("gateway unreachable")
        with pytest.raises(MouseOutputError):
            await mouse.move(1, 0)


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        mouse, _ = _mouse()
        mouse._connected = True
        await mouse.disconnect()
        assert mouse._connected is False
