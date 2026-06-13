# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/raspi; lint cleanup deferred.
"""Bluetooth HID combo device (keyboard + mouse) for Raspberry Pi.

Registers the Pi as a Bluetooth HID device using BlueZ and D-Bus.
Paired devices receive keyboard and mouse events over a single connection.

Uses Report IDs to multiplex keyboard and mouse reports on one L2CAP
interrupt channel:
  - Report ID 1: Keyboard (8 bytes: modifier, reserved, key1..key6)
  - Report ID 2: Mouse    (4 bytes: buttons, x_delta, y_delta, wheel)

L2CAP channels:
  - PSM 17 (0x11): Control channel
  - PSM 19 (0x13): Interrupt channel (HID reports sent here)
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
from enum import IntFlag

from handsneyes.pi.hid_codes import (
    char_to_hid,
    key_name_to_hid,
    modifiers_to_bitmask,
    MODIFIER_LEFT_SHIFT,
    MODIFIER_NONE,
    SHIFT_CHARS,
)

logger = logging.getLogger(__name__)

# L2CAP Protocol/Service Multiplexer values for HID
PSM_CONTROL = 0x11  # 17
PSM_INTERRUPT = 0x13  # 19

# Report IDs
REPORT_ID_KEYBOARD = 0x01
REPORT_ID_MOUSE = 0x02

# Default timing for keyboard events
DEFAULT_KEYPRESS_DELAY = 0.02
DEFAULT_INTER_CHAR_DELAY = 0.01

# Combined HID report descriptor: keyboard (ID 1) + mouse (ID 2)
COMBO_REPORT_DESCRIPTOR = bytes([
    # ===== Keyboard (Report ID 1) =====
    0x05, 0x01,        # Usage Page (Generic Desktop)
    0x09, 0x06,        # Usage (Keyboard)
    0xA1, 0x01,        # Collection (Application)
    0x85, REPORT_ID_KEYBOARD,  # Report ID (1)
    # Modifier keys (8 bits)
    0x05, 0x07,        #   Usage Page (Keyboard/Keypad)
    0x19, 0xE0,        #   Usage Minimum (Left Control)
    0x29, 0xE7,        #   Usage Maximum (Right Meta)
    0x15, 0x00,        #   Logical Minimum (0)
    0x25, 0x01,        #   Logical Maximum (1)
    0x75, 0x01,        #   Report Size (1)
    0x95, 0x08,        #   Report Count (8)
    0x81, 0x02,        #   Input (Data, Variable, Absolute)
    # Reserved byte
    0x95, 0x01,        #   Report Count (1)
    0x75, 0x08,        #   Report Size (8)
    0x81, 0x01,        #   Input (Constant)
    # Key codes (6 keys)
    0x95, 0x06,        #   Report Count (6)
    0x75, 0x08,        #   Report Size (8)
    0x15, 0x00,        #   Logical Minimum (0)
    0x25, 0x65,        #   Logical Maximum (101)
    0x05, 0x07,        #   Usage Page (Keyboard/Keypad)
    0x19, 0x00,        #   Usage Minimum (0)
    0x29, 0x65,        #   Usage Maximum (101)
    0x81, 0x00,        #   Input (Data, Array)
    0xC0,              # End Collection

    # ===== Mouse (Report ID 2) =====
    0x05, 0x01,        # Usage Page (Generic Desktop)
    0x09, 0x02,        # Usage (Mouse)
    0xA1, 0x01,        # Collection (Application)
    0x85, REPORT_ID_MOUSE,  # Report ID (2)
    0x09, 0x01,        #   Usage (Pointer)
    0xA1, 0x00,        #   Collection (Physical)
    # Buttons (3 buttons + 5 padding bits)
    0x05, 0x09,        #     Usage Page (Button)
    0x19, 0x01,        #     Usage Minimum (1)
    0x29, 0x03,        #     Usage Maximum (3)
    0x15, 0x00,        #     Logical Minimum (0)
    0x25, 0x01,        #     Logical Maximum (1)
    0x95, 0x03,        #     Report Count (3)
    0x75, 0x01,        #     Report Size (1)
    0x81, 0x02,        #     Input (Data, Variable, Absolute)
    0x95, 0x01,        #     Report Count (1)
    0x75, 0x05,        #     Report Size (5)
    0x81, 0x01,        #     Input (Constant) — padding
    # X, Y movement
    0x05, 0x01,        #     Usage Page (Generic Desktop)
    0x09, 0x30,        #     Usage (X)
    0x09, 0x31,        #     Usage (Y)
    0x15, 0x81,        #     Logical Minimum (-127)
    0x25, 0x7F,        #     Logical Maximum (127)
    0x75, 0x08,        #     Report Size (8)
    0x95, 0x02,        #     Report Count (2)
    0x81, 0x06,        #     Input (Data, Variable, Relative)
    # Scroll wheel
    0x09, 0x38,        #     Usage (Wheel)
    0x15, 0x81,        #     Logical Minimum (-127)
    0x25, 0x7F,        #     Logical Maximum (127)
    0x75, 0x08,        #     Report Size (8)
    0x95, 0x01,        #     Report Count (1)
    0x81, 0x06,        #     Input (Data, Variable, Relative)
    0xC0,              #   End Collection
    0xC0,              # End Collection
])

# SDP record XML for a Bluetooth HID combo device (keyboard + mouse).
SDP_RECORD_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<record>
  <attribute id="0x0001"> <!-- ServiceClassIDList -->
    <sequence>
      <uuid value="0x1124" /> <!-- HumanInterfaceDeviceService -->
    </sequence>
  </attribute>
  <attribute id="0x0004"> <!-- ProtocolDescriptorList -->
    <sequence>
      <sequence>
        <uuid value="0x0100" /> <!-- L2CAP -->
        <uint16 value="0x0011" /> <!-- PSM=HID_Control -->
      </sequence>
      <sequence>
        <uuid value="0x0011" /> <!-- HIDP -->
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x0005"> <!-- BrowseGroupList -->
    <sequence>
      <uuid value="0x1002" /> <!-- PublicBrowseRoot -->
    </sequence>
  </attribute>
  <attribute id="0x0006"> <!-- LanguageBaseAttributeIDList -->
    <sequence>
      <uint16 value="0x656E" /> <!-- en -->
      <uint16 value="0x006A" /> <!-- UTF-8 -->
      <uint16 value="0x0100" /> <!-- PrimaryLanguage -->
    </sequence>
  </attribute>
  <attribute id="0x0009"> <!-- BluetoothProfileDescriptorList -->
    <sequence>
      <sequence>
        <uuid value="0x1124" /> <!-- HumanInterfaceDeviceService -->
        <uint16 value="0x0101" /> <!-- Version 1.1 -->
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x000D"> <!-- AdditionalProtocolDescriptorList -->
    <sequence>
      <sequence>
        <sequence>
          <uuid value="0x0100" /> <!-- L2CAP -->
          <uint16 value="0x0013" /> <!-- PSM=HID_Interrupt -->
        </sequence>
        <sequence>
          <uuid value="0x0011" /> <!-- HIDP -->
        </sequence>
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x0100"> <!-- ServiceName -->
    <text value="devmouse" />
  </attribute>
  <attribute id="0x0101"> <!-- ServiceDescription -->
    <text value="Bluetooth Keyboard + Mouse (devmouse)" />
  </attribute>
  <attribute id="0x0102"> <!-- ProviderName -->
    <text value="handsneyes" />
  </attribute>
  <attribute id="0x0200"> <!-- HIDDeviceReleaseNumber -->
    <uint16 value="0x0100" />
  </attribute>
  <attribute id="0x0201"> <!-- HIDParserVersion -->
    <uint16 value="0x0111" />
  </attribute>
  <attribute id="0x0202"> <!-- HIDDeviceSubclass -->
    <uint8 value="0xC0" /> <!-- Combo: keyboard + pointing -->
  </attribute>
  <attribute id="0x0203"> <!-- HIDCountryCode -->
    <uint8 value="0x00" />
  </attribute>
  <attribute id="0x0204"> <!-- HIDVirtualCable -->
    <boolean value="true" />
  </attribute>
  <attribute id="0x0205"> <!-- HIDReconnectInitiate -->
    <boolean value="true" />
  </attribute>
  <attribute id="0x0206"> <!-- HIDDescriptorList -->
    <sequence>
      <sequence>
        <uint8 value="0x22" /> <!-- Report Descriptor -->
        <text encoding="hex" value="{report_desc_hex}" />
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x0207"> <!-- HIDLANGIDBaseList -->
    <sequence>
      <sequence>
        <uint16 value="0x0409" /> <!-- English (US) -->
        <uint16 value="0x0100" />
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x020B"> <!-- HIDProfileVersion -->
    <uint16 value="0x0100" />
  </attribute>
  <attribute id="0x020C"> <!-- HIDSupervisionTimeout -->
    <uint16 value="0x0C80" />
  </attribute>
  <attribute id="0x020D"> <!-- HIDNormallyConnectable -->
    <!-- CRITICAL for macOS auto-reconnect. When true, the host (Mac)
         knows this device is always in page-scan mode and can initiate
         the HID L2CAP channels itself when it detects the Pi nearby —
         exactly what a Logitech BT mouse does. Without this the Mac
         waits for the Pi to initiate, but macOS blocks Pi-initiated HID
         L2CAP (the "BR/EDR up but HID not" deadlock in the watchdog). -->
    <boolean value="true" />
  </attribute>
  <attribute id="0x020E"> <!-- HIDBootDevice -->
    <boolean value="true" />
  </attribute>
</record>
"""


class MouseButton(IntFlag):
    """Mouse button bitmask for the HID report."""
    NONE = 0x00
    LEFT = 0x01
    RIGHT = 0x02
    MIDDLE = 0x04


BUTTON_MAP: dict[str, MouseButton] = {
    "left": MouseButton.LEFT,
    "right": MouseButton.RIGHT,
    "middle": MouseButton.MIDDLE,
}


class BtHidError(Exception):
    """Raised when Bluetooth HID operations fail."""


class _HidClient:
    """One connected host: its socket pair + per-host input state."""

    __slots__ = ("mac", "ctrl", "intr", "mouse_buttons", "control_task")

    def __init__(
        self, mac: str, ctrl: socket.socket, intr: socket.socket,
    ) -> None:
        self.mac = mac
        self.ctrl = ctrl
        self.intr = intr
        self.mouse_buttons: int = 0
        self.control_task: asyncio.Task[None] | None = None


def _clamp(value: int, minimum: int = -127, maximum: int = 127) -> int:
    return max(minimum, min(maximum, value))


def build_sdp_record() -> str:
    """Build the SDP record XML with the combo report descriptor."""
    return SDP_RECORD_XML.format(report_desc_hex=COMBO_REPORT_DESCRIPTOR.hex())


# 8-byte keyboard release report (all zeros)
_KB_RELEASE = bytes(8)


class BluetoothHidServer:
    """Bluetooth HID combo device (keyboard + mouse) over L2CAP.

    Listens on L2CAP PSM 17 (control) and PSM 19 (interrupt), waits for
    a Bluetooth host to connect, then sends keyboard and mouse HID
    reports on the interrupt channel.

    Each report is prefixed with 0xA1 (HIDP DATA|INPUT header) followed
    by the report ID and report data.

    The control channel (PSM 17) is monitored for HIDP protocol messages
    like SET_PROTOCOL; responses are sent automatically.

    Multi-host: any number of bonded hosts may be connected at once,
    each with its own socket pair and input state. Every send targets
    exactly ONE host — explicit ``host=`` MAC, else the active host,
    else the single connection. Other hosts never receive a byte.

    Usage::

        server = BluetoothHidServer()
        await server.start()
        asyncio.create_task(server.accept_forever())

        # Keyboard (explicit host, or active/single fallback)
        await server.send_keystroke("Enter", host="84:2F:57:7D:85:21")
        await server.send_key_combo(["ctrl"], "c")
        await server.send_text("hello")

        # Mouse
        await server.move(10, -5)
        await server.click("left")
        await server.scroll(-3)

        await server.stop()
    """

    # HIDP transaction types (high nibble of first byte)
    _HIDP_HANDSHAKE = 0x00
    _HIDP_HID_CONTROL = 0x10
    _HIDP_GET_REPORT = 0x40
    _HIDP_SET_REPORT = 0x50
    _HIDP_GET_PROTOCOL = 0x60
    _HIDP_SET_PROTOCOL = 0x70

    # HIDP handshake parameters
    _HANDSHAKE_SUCCESS = 0x00
    _HANDSHAKE_NOT_READY = 0x01
    _HANDSHAKE_ERR_UNSUPPORTED = 0x05

    def __init__(
        self,
        keypress_delay: float = DEFAULT_KEYPRESS_DELAY,
        inter_char_delay: float = DEFAULT_INTER_CHAR_DELAY,
    ) -> None:
        self._keypress_delay = keypress_delay
        self._inter_char_delay = inter_char_delay
        self._control_sock: socket.socket | None = None
        self._interrupt_sock: socket.socket | None = None
        # Multi-host: every connected host gets its own client record
        # keyed by MAC, with its own socket pair and its own mouse-
        # button state (a drag held on host A must not leak into a
        # report sent to host B). Reports go ONLY to the addressed
        # host — the others never see a byte.
        self._clients: dict[str, _HidClient] = {}
        # Half-open connections: a host opens the control channel
        # first, then the interrupt channel. With several hosts
        # connecting concurrently the accepts can interleave, so the
        # two accept loops pair sockets by peer MAC here.
        self._pending: dict[str, dict[str, socket.socket]] = {}
        self._active_mac: str | None = None
        self._stopping = False

    @property
    def is_connected(self) -> bool:
        return bool(self._clients)

    @property
    def connected_hosts(self) -> list[str]:
        return list(self._clients)

    @property
    def active_host(self) -> str | None:
        return self._active_mac

    def set_active_host(self, mac: str) -> None:
        mac = mac.strip().upper()
        if mac not in self._clients:
            raise BtHidError(f"Host {mac} is not connected")
        self._active_mac = mac

    def _resolve(self, host: str | None) -> "_HidClient":
        """Pick the client a report should go to.

        Explicit ``host`` wins. Otherwise the active host, otherwise
        the single connected host. Ambiguity (several hosts, no
        active, no explicit host) is an error — silently picking one
        would type into somebody's machine."""
        if host:
            mac = host.strip().upper()
            client = self._clients.get(mac)
            if client is None:
                raise BtHidError(
                    f"Host {mac} is not connected "
                    f"(connected: {sorted(self._clients) or 'none'})"
                )
            return client
        if self._active_mac is not None:
            client = self._clients.get(self._active_mac)
            if client is not None:
                return client
        if len(self._clients) == 1:
            return next(iter(self._clients.values()))
        if not self._clients:
            raise BtHidError("No Bluetooth client connected")
        raise BtHidError(
            "Multiple hosts connected and no active host set — "
            "pass 'host' or POST /bt/active-host first "
            f"(connected: {sorted(self._clients)})"
        )

    async def start(self) -> None:
        """Open L2CAP listening sockets."""
        try:
            self._control_sock = socket.socket(
                socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP
            )
            self._interrupt_sock = socket.socket(
                socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP
            )
            self._control_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._interrupt_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._control_sock.bind(("00:00:00:00:00:00", PSM_CONTROL))
            self._interrupt_sock.bind(("00:00:00:00:00:00", PSM_INTERRUPT))
            # Backlog > 1: several hosts may (re)connect concurrently.
            self._control_sock.listen(4)
            self._interrupt_sock.listen(4)
            logger.info(
                "Bluetooth HID server listening (PSM %d control, PSM %d interrupt)",
                PSM_CONTROL, PSM_INTERRUPT,
            )
        except OSError as e:
            raise BtHidError(f"Failed to create L2CAP sockets: {e}") from e

    async def _control_channel_loop(self, client: "_HidClient") -> None:
        """Read and respond to HIDP messages on one host's control
        channel. Runs as a background task per connected host. Handles
        SET_PROTOCOL (0x70/0x71) and GET_PROTOCOL (0x60) which macOS
        sends during HID connection setup. EOF means the host
        disconnected — only THAT host's record is dropped.
        """
        sock = client.ctrl
        loop = asyncio.get_running_loop()
        sock.setblocking(False)
        try:
            while client.mac in self._clients:
                try:
                    data = await loop.sock_recv(sock, 1024)
                except (BlockingIOError, OSError):
                    break
                if not data:
                    logger.info(
                        "Control channel closed by %s", client.mac,
                    )
                    break
                msg_type = data[0] & 0xF0
                param = data[0] & 0x0F
                logger.info(
                    "Control channel msg: 0x%02X (type=0x%02X param=0x%02X) %s",
                    data[0], msg_type, param, data.hex(),
                )
                if msg_type == self._HIDP_SET_PROTOCOL:
                    # param: 0=Boot Protocol, 1=Report Protocol
                    logger.info(
                        "SET_PROTOCOL: %s mode",
                        "Report" if param == 1 else "Boot",
                    )
                    await loop.sock_sendall(
                        sock, bytes([self._HANDSHAKE_SUCCESS])
                    )
                elif msg_type == self._HIDP_GET_PROTOCOL:
                    # Respond with Report Protocol (0x01)
                    await loop.sock_sendall(sock, bytes([0x01]))
                elif msg_type == self._HIDP_SET_REPORT:
                    # ACK output reports (e.g. LED state)
                    await loop.sock_sendall(
                        sock, bytes([self._HANDSHAKE_SUCCESS])
                    )
                elif msg_type == self._HIDP_HID_CONTROL:
                    if param == 0x03:  # EXIT_SUSPEND
                        logger.info("HID_CONTROL: exit suspend")
                    else:
                        logger.info("HID_CONTROL: param=0x%02X", param)
                else:
                    logger.info("Unhandled control msg type 0x%02X", msg_type)
        except Exception as e:
            logger.debug(
                "Control channel loop for %s ended: %s", client.mac, e,
            )
        finally:
            # Whatever ended the loop, this host is gone.
            self._drop_client(client.mac, "control channel ended")

    async def accept_forever(self) -> None:
        """Accept connections from any number of hosts, forever.

        Runs one accept loop per listening socket; control and
        interrupt sockets are paired by peer MAC (concurrent hosts
        can interleave their channel opens)."""
        if not self._control_sock or not self._interrupt_sock:
            raise BtHidError("Server not started")
        logger.info("Waiting for Bluetooth HID connections (multi-host)...")
        await asyncio.gather(
            self._accept_loop(self._control_sock, "ctrl"),
            self._accept_loop(self._interrupt_sock, "intr"),
        )

    async def _accept_loop(self, lsock: socket.socket, kind: str) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopping:
            try:
                sock, addr = await loop.run_in_executor(None, lsock.accept)
            except OSError:
                return  # listener closed (stop())
            mac = str(addr[0]).upper()
            logger.info("%s channel connected from %s", kind, mac)
            pend = self._pending.setdefault(mac, {})
            old = pend.get(kind)
            if old is not None:
                try:
                    old.close()
                except OSError:
                    pass
            pend[kind] = sock
            if "ctrl" in pend and "intr" in pend:
                self._pending.pop(mac, None)
                self._promote(mac, pend["ctrl"], pend["intr"])

    def _promote(
        self, mac: str, ctrl: socket.socket, intr: socket.socket,
    ) -> None:
        stale = self._clients.pop(mac, None)
        if stale is not None:
            self._close_client(stale)
        client = _HidClient(mac, ctrl, intr)
        self._clients[mac] = client
        client.control_task = asyncio.create_task(
            self._control_channel_loop(client),
        )
        if self._active_mac is None:
            self._active_mac = mac
        logger.warning(
            "Bluetooth HID host connected: %s (%d connected: %s; "
            "active: %s)",
            mac, len(self._clients), sorted(self._clients),
            self._active_mac,
        )

    def _close_client(self, client: "_HidClient") -> None:
        task = client.control_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        for sock in (client.intr, client.ctrl):
            try:
                sock.close()
            except OSError:
                pass

    def _drop_client(self, mac: str, reason: str) -> None:
        client = self._clients.pop(mac, None)
        if client is None:
            return
        self._close_client(client)
        if self._active_mac == mac:
            self._active_mac = next(iter(self._clients), None)
        logger.warning(
            "Bluetooth HID host disconnected: %s (%s; %d remain: %s; "
            "active: %s)",
            mac, reason, len(self._clients), sorted(self._clients),
            self._active_mac,
        )

    async def _send_raw(
        self, data: bytes, client: "_HidClient",
    ) -> None:
        """Send raw bytes on ONE host's interrupt channel."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, client.intr.send, data)
        except OSError as e:
            self._drop_client(client.mac, f"send failed: {e}")
            raise BtHidError(
                f"Failed to send HID report to {client.mac}: {e}",
            ) from e

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    async def _send_keyboard_report(
        self, modifier: int, scan_code: int, client: "_HidClient",
    ) -> None:
        """Send a keyboard HID report (report ID 1)."""
        # 0xA1 = HIDP DATA|INPUT, then report ID, then 8 bytes keyboard
        report = bytes([
            0xA1,
            REPORT_ID_KEYBOARD,
            modifier, 0x00, scan_code, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        await self._send_raw(report, client)

    async def _release_keyboard(self, client: "_HidClient") -> None:
        """Send an all-zeros keyboard report (release all keys)."""
        report = bytes([0xA1, REPORT_ID_KEYBOARD]) + _KB_RELEASE
        await self._send_raw(report, client)

    async def _tap_key(
        self, modifier: int, scan_code: int, client: "_HidClient",
    ) -> None:
        """Press and release a key with timing."""
        await self._send_keyboard_report(modifier, scan_code, client)
        await asyncio.sleep(self._keypress_delay)
        await self._release_keyboard(client)

    async def _keystroke_preflight(self, client: "_HidClient") -> None:
        """Clear lingering keyboard state and let the receiver's
        input subsystem drain before sending a new key.

        Without this, a key sent immediately after a prior burst
        (e.g. the Enter that follows send_text, or the first key
        of the next agent step) lands during the kernel's busy
        window and is silently dropped — same root cause as the
        first-character-of-text drop. Costs ~350ms per single
        keystroke, which is acceptable for tap-style calls
        (Enter / Esc / Tab / chord). send_text bypasses this
        because its inter-char loop has its own one-shot
        pre-flight at the top.
        """
        for _ in range(2):
            try:
                await self._release_keyboard(client)
            except Exception:
                pass
            await asyncio.sleep(0.08)
        await asyncio.sleep(0.15)

    async def send_keystroke(
        self, key: str, *, host: str | None = None,
    ) -> None:
        """Send a named key (e.g., 'Enter', 'Tab', 'a')."""
        client = self._resolve(host)
        if key in SHIFT_CHARS:
            modifier, scan_code = char_to_hid(key)
        elif len(key) == 1:
            modifier, scan_code = char_to_hid(key)
        else:
            scan_code = key_name_to_hid(key)
            modifier = MODIFIER_NONE
        await self._keystroke_preflight(client)
        await self._tap_key(modifier, scan_code, client)
        logger.debug(
            "BT keystroke → %s: %s (mod=0x%02X scan=0x%02X)",
            client.mac, key, modifier, scan_code,
        )

    async def send_key_combo(
        self, modifiers: list[str], key: str, *, host: str | None = None,
    ) -> None:
        """Send a key combination (e.g., ctrl+c)."""
        client = self._resolve(host)
        mod_bitmask = modifiers_to_bitmask(modifiers)
        if key in SHIFT_CHARS:
            base_char = SHIFT_CHARS[key]
            scan_code = key_name_to_hid(base_char)
            mod_bitmask |= MODIFIER_LEFT_SHIFT
        else:
            scan_code = key_name_to_hid(key)
        await self._keystroke_preflight(client)
        await self._tap_key(mod_bitmask, scan_code, client)
        logger.debug(
            "BT combo → %s: %s+%s (mod=0x%02X scan=0x%02X)",
            client.mac, "+".join(modifiers), key, mod_bitmask, scan_code,
        )

    async def send_text(
        self, text: str, *, warmup: bool = True, host: str | None = None,
    ) -> None:
        """Type a string character by character.

        Pre-flight (defensive against the receiving end's input
        buffer being busy after a prior keystroke — e.g. Enter
        from a launch, profile-picker dance, etc.):

          1. Three release reports (spaced 100ms apart) drain any
             lingering keyboard state and give the kernel + the
             foreground app time to consume all of them.
          2. A 500ms settle BEFORE the first real character.

        We previously tried a modifier-only "warm-up" tap (Shift
        alone with scan=0) to nudge the receiver's input pipeline
        awake; on some bluez stacks the receiver interpreted that
        report as Escape (visible as ``^[`` prefixed to typed
        text). The plain settle is slower but produces no
        visible artifact.

        Without this pre-flight the first character of ``text``
        lands during the kernel's busy window and is silently
        dropped. Observed symptoms: ``echo hello`` → ``cho hello``,
        ``uname -r`` → ``name -r``, ``clear`` → ``lear`` — always
        exactly one missing leading character.
        """
        if not text:
            return
        client = self._resolve(host)
        # Pre-flight: three releases + 500ms settle. Drains any
        # half-processed report state on the receiver side.
        for _ in range(3):
            try:
                await self._release_keyboard(client)
            except Exception:
                pass
            await asyncio.sleep(0.10)
        await asyncio.sleep(0.50)

        # First-character warmup. The receiver's input subsystem
        # reliably drops the FIRST keypress after an idle period.
        # We type the first char TWICE with a Backspace between,
        # so:
        #   - First press registered → "X<BS>X" → "X"
        #   - First press eaten      → "<BS>X"   → "X"
        # Either way, exactly one instance of the intended first
        # character lands.
        #
        # Caveat: in some browser URL bars (e.g. Firefox), the
        # Backspace key is bound to history-back rather than
        # character deletion. There, this warmup produces a
        # doubled first character — pass ``warmup=False`` from
        # callers that target such contexts (NavigateAgent does
        # this when typing into a browser URL bar). Without
        # warmup the first character occasionally drops, but
        # the caller can detect that via post-OCR and retry.
        if warmup:
            first_mod, first_scan = char_to_hid(text[0])
            bs_scan = key_name_to_hid("Backspace")
            await self._tap_key(first_mod, first_scan, client)
            await asyncio.sleep(self._inter_char_delay)
            await self._tap_key(MODIFIER_NONE, bs_scan, client)
            await asyncio.sleep(self._inter_char_delay)
            await self._tap_key(first_mod, first_scan, client)
            await asyncio.sleep(self._inter_char_delay)

            chars_iter = text[1:]
        else:
            chars_iter = text

        for char in chars_iter:
            modifier, scan_code = char_to_hid(char)
            await self._tap_key(modifier, scan_code, client)
            await asyncio.sleep(self._inter_char_delay)
        logger.debug("BT text → %s: %s", client.mac, text[:50])

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    async def _send_mouse_report(
        self, buttons: int, x: int, y: int, wheel: int,
        client: "_HidClient",
    ) -> None:
        """Send a mouse HID report (report ID 2)."""
        x = _clamp(x)
        y = _clamp(y)
        wheel = _clamp(wheel)
        # 0xA1 header + report ID 2 + 4 bytes mouse data
        # buttons is unsigned byte, x/y are signed, wheel is signed
        report = struct.pack("BBBbbb", 0xA1, REPORT_ID_MOUSE, buttons, x, y, wheel)
        await self._send_raw(report, client)

    async def move(
        self, x: int, y: int, *, host: str | None = None,
    ) -> None:
        """Move the mouse cursor by (x, y) relative pixels."""
        client = self._resolve(host)
        await self._send_mouse_report(client.mouse_buttons, x, y, 0, client)
        logger.debug("BT mouse move → %s: dx=%d dy=%d", client.mac, x, y)

    async def click(
        self, button: str = "left", *, host: str | None = None,
    ) -> None:
        """Click a mouse button (press and release)."""
        await self.press(button, host=host)
        await asyncio.sleep(0.05)
        await self.release(button, host=host)
        logger.debug("BT mouse click: %s", button)

    async def press(
        self, button: str = "left", *, host: str | None = None,
    ) -> None:
        """Hold a mouse button down. Stays pressed until release(); used
        by drag-and-drop where moves between press/release become drag
        deltas instead of cursor-only motion. Button state is PER HOST
        — a drag held on one host never leaks into another's reports."""
        btn = BUTTON_MAP.get(button.lower())
        if btn is None:
            raise ValueError(f"Unknown button: {button!r}. Use: left, right, middle")
        client = self._resolve(host)
        client.mouse_buttons |= btn
        await self._send_mouse_report(
            client.mouse_buttons, 0, 0, 0, client,
        )
        logger.debug("BT mouse press → %s: %s", client.mac, button)

    async def release(
        self, button: str = "left", *, host: str | None = None,
    ) -> None:
        """Release a previously-pressed mouse button. Idempotent —
        releasing an already-released button just resends the current
        button state (cheap no-op)."""
        btn = BUTTON_MAP.get(button.lower())
        if btn is None:
            raise ValueError(f"Unknown button: {button!r}. Use: left, right, middle")
        client = self._resolve(host)
        client.mouse_buttons &= ~btn
        await self._send_mouse_report(
            client.mouse_buttons, 0, 0, 0, client,
        )
        logger.debug("BT mouse release → %s: %s", client.mac, button)

    async def scroll(
        self, amount: int, *, host: str | None = None,
    ) -> None:
        """Scroll the mouse wheel. Positive=up, negative=down."""
        client = self._resolve(host)
        await self._send_mouse_report(
            client.mouse_buttons, 0, 0, amount, client,
        )
        logger.debug("BT mouse scroll → %s: %d", client.mac, amount)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Close all sockets and cancel background tasks."""
        self._stopping = True
        for mac in list(self._clients):
            self._drop_client(mac, "server stopping")
        for pend in self._pending.values():
            for sock in pend.values():
                try:
                    sock.close()
                except OSError:
                    pass
        self._pending.clear()
        for sock in (self._interrupt_sock, self._control_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._interrupt_sock = None
        self._control_sock = None
        logger.info("Bluetooth HID server stopped")


# ------------------------------------------------------------------
# BlueZ / D-Bus helpers
# ------------------------------------------------------------------

def register_sdp_profile() -> None:
    """Register the HID combo SDP profile with BlueZ via D-Bus."""
    try:
        import dbus  # type: ignore[import-untyped]
    except ImportError as e:
        raise BtHidError(
            "python3-dbus not installed. Run: sudo apt install python3-dbus"
        ) from e

    bus = dbus.SystemBus()
    manager = dbus.Interface(
        bus.get_object("org.bluez", "/org/bluez"),
        "org.bluez.ProfileManager1",
    )

    opts = {
        "Role": "server",
        "RequireAuthentication": False,
        "RequireAuthorization": False,
        "AutoConnect": True,
        "ServiceRecord": build_sdp_record(),
    }

    try:
        manager.RegisterProfile(
            "/org/bluez/handsneyes_hid",
            "00001124-0000-1000-8000-00805f9b34fb",  # HID UUID
            opts,
        )
        logger.info("Bluetooth HID combo profile registered with BlueZ")
    except dbus.exceptions.DBusException as e:
        err_str = str(e)
        if "AlreadyExists" in err_str or "already registered" in err_str.lower():
            logger.info("Bluetooth HID profile already registered")
        elif "NotPermitted" in err_str:
            logger.info("Bluetooth HID profile registration not permitted (may already be active)")
        else:
            raise BtHidError(f"Failed to register BT profile: {e}") from e


def configure_bluetooth_adapter() -> None:
    """Make the Bluetooth adapter discoverable and set combo device class.

    Safe to call multiple times — silently ignores properties that are
    already set or that BlueZ refuses to change.
    """
    try:
        import dbus  # type: ignore[import-untyped]
    except ImportError as e:
        raise BtHidError(
            "python3-dbus not installed. Run: sudo apt install python3-dbus"
        ) from e

    bus = dbus.SystemBus()
    adapter = dbus.Interface(
        bus.get_object("org.bluez", "/org/bluez/hci0"),
        "org.freedesktop.DBus.Properties",
    )

    for prop, val in [
        ("Powered", dbus.Boolean(True)),
        ("Discoverable", dbus.Boolean(True)),
        ("DiscoverableTimeout", dbus.UInt32(0)),
        ("Pairable", dbus.Boolean(True)),
    ]:
        try:
            adapter.Set("org.bluez.Adapter1", prop, val)
        except dbus.exceptions.DBusException as e:
            logger.debug("Could not set adapter %s: %s (may already be set)", prop, e)

    logger.info("Bluetooth adapter configured: discoverable + pairable")
