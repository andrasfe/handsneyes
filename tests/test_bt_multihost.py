"""Multi-host BT HID routing (resolution + isolation semantics)."""

from __future__ import annotations

import pytest

from handsneyes.pi.bt_hid import BluetoothHidServer, BtHidError, _HidClient


class _FakeSock:
    def close(self) -> None:
        pass


def _server_with(*macs: str) -> BluetoothHidServer:
    srv = BluetoothHidServer()
    for mac in macs:
        srv._clients[mac] = _HidClient(mac, _FakeSock(), _FakeSock())
    return srv


def test_resolve_no_clients_raises() -> None:
    srv = _server_with()
    with pytest.raises(BtHidError, match="No Bluetooth client"):
        srv._resolve(None)


def test_resolve_single_client_is_default() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01")
    assert srv._resolve(None).mac == "AA:BB:CC:DD:EE:01"


def test_resolve_explicit_host_wins_and_normalizes() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02")
    srv._active_mac = "AA:BB:CC:DD:EE:01"
    assert srv._resolve(" aa:bb:cc:dd:ee:02 ").mac == "AA:BB:CC:DD:EE:02"


def test_resolve_unknown_host_raises() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01")
    with pytest.raises(BtHidError, match="not connected"):
        srv._resolve("AA:BB:CC:DD:EE:99")


def test_resolve_ambiguous_without_active_raises() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02")
    srv._active_mac = None
    with pytest.raises(BtHidError, match="Multiple hosts"):
        srv._resolve(None)


def test_active_host_routes_unaddressed() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02")
    srv.set_active_host("aa:bb:cc:dd:ee:02")
    assert srv._resolve(None).mac == "AA:BB:CC:DD:EE:02"


def test_set_active_host_requires_connected() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01")
    with pytest.raises(BtHidError):
        srv.set_active_host("AA:BB:CC:DD:EE:99")


def test_drop_client_reassigns_active() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02")
    srv._active_mac = "AA:BB:CC:DD:EE:01"
    srv._drop_client("AA:BB:CC:DD:EE:01", "test")
    assert srv.connected_hosts == ["AA:BB:CC:DD:EE:02"]
    assert srv.active_host == "AA:BB:CC:DD:EE:02"


def test_mouse_button_state_is_per_host() -> None:
    srv = _server_with("AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02")
    a = srv._resolve("AA:BB:CC:DD:EE:01")
    b = srv._resolve("AA:BB:CC:DD:EE:02")
    a.mouse_buttons |= 1
    assert b.mouse_buttons == 0
