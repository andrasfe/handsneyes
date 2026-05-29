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

    Usage::

        server = BluetoothHidServer()
        await server.start()
        addr = await server.wait_for_connection()

        # Keyboard
        await server.send_keystroke("Enter")
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
        self._control_client: socket.socket | None = None
        self._interrupt_client: socket.socket | None = None
        self._connected = False
        self._mouse_buttons: int = 0
        self._control_task: asyncio.Task[None] | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

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
            self._control_sock.listen(1)
            self._interrupt_sock.listen(1)
            logger.info(
                "Bluetooth HID server listening (PSM %d control, PSM %d interrupt)",
                PSM_CONTROL, PSM_INTERRUPT,
            )
        except OSError as e:
            raise BtHidError(f"Failed to create L2CAP sockets: {e}") from e

    async def _control_channel_loop(self) -> None:
        """Read and respond to HIDP messages on the control channel.

        Runs as a background task while a client is connected.  Handles
        SET_PROTOCOL (0x70/0x71) and GET_PROTOCOL (0x60) which macOS
        sends during HID connection setup.
        """
        sock = self._control_client
        if sock is None:
            return
        loop = asyncio.get_running_loop()
        sock.setblocking(False)
        try:
            while self._connected:
                try:
                    data = await loop.sock_recv(sock, 1024)
                except (BlockingIOError, OSError):
                    break
                if not data:
                    logger.info("Control channel closed by peer")
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
            logger.debug("Control channel loop ended: %s", e)

    async def wait_for_connection(self) -> str:
        """Wait for a Bluetooth host to connect. Returns host address."""
        if not self._control_sock or not self._interrupt_sock:
            raise BtHidError("Server not started")

        loop = asyncio.get_running_loop()
        logger.info("Waiting for Bluetooth HID connection...")

        self._control_client, ctrl_addr = await loop.run_in_executor(
            None, self._control_sock.accept
        )
        logger.info("Control channel connected from %s", ctrl_addr[0])

        self._interrupt_client, intr_addr = await loop.run_in_executor(
            None, self._interrupt_sock.accept
        )
        logger.info("Interrupt channel connected from %s", intr_addr[0])

        self._connected = True

        # Start reading control channel messages in the background
        self._control_task = asyncio.create_task(self._control_channel_loop())

        return ctrl_addr[0]

    async def _send_raw(self, data: bytes) -> None:
        """Send raw bytes on the interrupt channel."""
        if not self._interrupt_client:
            raise BtHidError("No Bluetooth client connected")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._interrupt_client.send, data)
        except OSError as e:
            self._connected = False
            raise BtHidError(f"Failed to send HID report: {e}") from e

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    async def _send_keyboard_report(self, modifier: int, scan_code: int) -> None:
        """Send a keyboard HID report (report ID 1)."""
        # 0xA1 = HIDP DATA|INPUT, then report ID, then 8 bytes keyboard
        report = bytes([
            0xA1,
            REPORT_ID_KEYBOARD,
            modifier, 0x00, scan_code, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        await self._send_raw(report)

    async def _release_keyboard(self) -> None:
        """Send an all-zeros keyboard report (release all keys)."""
        report = bytes([0xA1, REPORT_ID_KEYBOARD]) + _KB_RELEASE
        await self._send_raw(report)

    async def _tap_key(self, modifier: int, scan_code: int) -> None:
        """Press and release a key with timing."""
        await self._send_keyboard_report(modifier, scan_code)
        await asyncio.sleep(self._keypress_delay)
        await self._release_keyboard()

    async def _keystroke_preflight(self) -> None:
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
                await self._release_keyboard()
            except Exception:
                pass
            await asyncio.sleep(0.08)
        await asyncio.sleep(0.15)

    async def send_keystroke(self, key: str) -> None:
        """Send a named key (e.g., 'Enter', 'Tab', 'a')."""
        if key in SHIFT_CHARS:
            modifier, scan_code = char_to_hid(key)
        elif len(key) == 1:
            modifier, scan_code = char_to_hid(key)
        else:
            scan_code = key_name_to_hid(key)
            modifier = MODIFIER_NONE
        await self._keystroke_preflight()
        await self._tap_key(modifier, scan_code)
        logger.debug("BT keystroke: %s (mod=0x%02X scan=0x%02X)", key, modifier, scan_code)

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination (e.g., ctrl+c)."""
        mod_bitmask = modifiers_to_bitmask(modifiers)
        if key in SHIFT_CHARS:
            base_char = SHIFT_CHARS[key]
            scan_code = key_name_to_hid(base_char)
            mod_bitmask |= MODIFIER_LEFT_SHIFT
        else:
            scan_code = key_name_to_hid(key)
        await self._keystroke_preflight()
        await self._tap_key(mod_bitmask, scan_code)
        logger.debug(
            "BT combo: %s+%s (mod=0x%02X scan=0x%02X)",
            "+".join(modifiers), key, mod_bitmask, scan_code,
        )

    async def send_text(self, text: str, *, warmup: bool = True) -> None:
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
        # Pre-flight: three releases + 500ms settle. Drains any
        # half-processed report state on the receiver side.
        for _ in range(3):
            try:
                await self._release_keyboard()
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
            await self._tap_key(first_mod, first_scan)
            await asyncio.sleep(self._inter_char_delay)
            await self._tap_key(MODIFIER_NONE, bs_scan)
            await asyncio.sleep(self._inter_char_delay)
            await self._tap_key(first_mod, first_scan)
            await asyncio.sleep(self._inter_char_delay)

            chars_iter = text[1:]
        else:
            chars_iter = text

        for char in chars_iter:
            modifier, scan_code = char_to_hid(char)
            await self._tap_key(modifier, scan_code)
            await asyncio.sleep(self._inter_char_delay)
        logger.debug("BT text: %s", text[:50])

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    async def _send_mouse_report(
        self, buttons: int, x: int, y: int, wheel: int
    ) -> None:
        """Send a mouse HID report (report ID 2)."""
        x = _clamp(x)
        y = _clamp(y)
        wheel = _clamp(wheel)
        # 0xA1 header + report ID 2 + 4 bytes mouse data
        # buttons is unsigned byte, x/y are signed, wheel is signed
        report = struct.pack("BBBbbb", 0xA1, REPORT_ID_MOUSE, buttons, x, y, wheel)
        await self._send_raw(report)

    async def move(self, x: int, y: int) -> None:
        """Move the mouse cursor by (x, y) relative pixels."""
        await self._send_mouse_report(self._mouse_buttons, x, y, 0)
        logger.debug("BT mouse move: dx=%d dy=%d", x, y)

    async def click(self, button: str = "left") -> None:
        """Click a mouse button (press and release)."""
        await self.press(button)
        await asyncio.sleep(0.05)
        await self.release(button)
        logger.debug("BT mouse click: %s", button)

    async def press(self, button: str = "left") -> None:
        """Hold a mouse button down. Stays pressed until release(); used
        by drag-and-drop where moves between press/release become drag
        deltas instead of cursor-only motion."""
        btn = BUTTON_MAP.get(button.lower())
        if btn is None:
            raise ValueError(f"Unknown button: {button!r}. Use: left, right, middle")
        self._mouse_buttons |= btn
        await self._send_mouse_report(self._mouse_buttons, 0, 0, 0)
        logger.debug("BT mouse press: %s", button)

    async def release(self, button: str = "left") -> None:
        """Release a previously-pressed mouse button. Idempotent —
        releasing an already-released button just resends the current
        button state (cheap no-op)."""
        btn = BUTTON_MAP.get(button.lower())
        if btn is None:
            raise ValueError(f"Unknown button: {button!r}. Use: left, right, middle")
        self._mouse_buttons &= ~btn
        await self._send_mouse_report(self._mouse_buttons, 0, 0, 0)
        logger.debug("BT mouse release: %s", button)

    async def scroll(self, amount: int) -> None:
        """Scroll the mouse wheel. Positive=up, negative=down."""
        await self._send_mouse_report(self._mouse_buttons, 0, 0, amount)
        logger.debug("BT mouse scroll: %d", amount)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Close all sockets and cancel background tasks."""
        self._connected = False
        if self._control_task is not None:
            self._control_task.cancel()
            self._control_task = None
        for sock in (
            self._interrupt_client,
            self._control_client,
            self._interrupt_sock,
            self._control_sock,
        ):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._interrupt_client = None
        self._control_client = None
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
