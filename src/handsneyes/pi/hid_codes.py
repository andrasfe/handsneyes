# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/raspi; lint cleanup deferred.
"""USB HID keyboard scan codes and modifier bitmasks.

Reference: USB HID Usage Tables v1.4, Section 10 (Keyboard/Keypad Page 0x07).

A USB HID keyboard report is 8 bytes:
    [modifier_byte, 0x00, key1, key2, key3, key4, key5, key6]

- Byte 0: modifier bitmask (ctrl, shift, alt, meta for left/right)
- Byte 1: reserved (always 0x00)
- Bytes 2-7: up to 6 simultaneous key scan codes
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Modifier bitmasks (byte 0 of HID report)
# ---------------------------------------------------------------------------

MODIFIER_NONE: int = 0x00
MODIFIER_LEFT_CTRL: int = 0x01
MODIFIER_LEFT_SHIFT: int = 0x02
MODIFIER_LEFT_ALT: int = 0x04
MODIFIER_LEFT_META: int = 0x08
MODIFIER_RIGHT_CTRL: int = 0x10
MODIFIER_RIGHT_SHIFT: int = 0x20
MODIFIER_RIGHT_ALT: int = 0x40
MODIFIER_RIGHT_META: int = 0x80

# Friendly names -> modifier bitmask
MODIFIER_MAP: dict[str, int] = {
    "ctrl": MODIFIER_LEFT_CTRL,
    "left_ctrl": MODIFIER_LEFT_CTRL,
    "right_ctrl": MODIFIER_RIGHT_CTRL,
    "shift": MODIFIER_LEFT_SHIFT,
    "left_shift": MODIFIER_LEFT_SHIFT,
    "right_shift": MODIFIER_RIGHT_SHIFT,
    "alt": MODIFIER_LEFT_ALT,
    "left_alt": MODIFIER_LEFT_ALT,
    "right_alt": MODIFIER_RIGHT_ALT,
    "meta": MODIFIER_LEFT_META,
    "super": MODIFIER_LEFT_META,
    "win": MODIFIER_LEFT_META,
    "left_meta": MODIFIER_LEFT_META,
    "right_meta": MODIFIER_RIGHT_META,
}

# ---------------------------------------------------------------------------
# Key name -> USB HID scan code
# ---------------------------------------------------------------------------

KEY_CODES: dict[str, int] = {
    # Letters (a=0x04 .. z=0x1D)
    "a": 0x04, "b": 0x05, "c": 0x06, "d": 0x07,
    "e": 0x08, "f": 0x09, "g": 0x0A, "h": 0x0B,
    "i": 0x0C, "j": 0x0D, "k": 0x0E, "l": 0x0F,
    "m": 0x10, "n": 0x11, "o": 0x12, "p": 0x13,
    "q": 0x14, "r": 0x15, "s": 0x16, "t": 0x17,
    "u": 0x18, "v": 0x19, "w": 0x1A, "x": 0x1B,
    "y": 0x1C, "z": 0x1D,
    # Numbers (1=0x1E .. 0=0x27)
    "1": 0x1E, "2": 0x1F, "3": 0x20, "4": 0x21,
    "5": 0x22, "6": 0x23, "7": 0x24, "8": 0x25,
    "9": 0x26, "0": 0x27,
    # Control keys
    "Enter": 0x28, "Return": 0x28,
    "Escape": 0x29, "Esc": 0x29,
    "Backspace": 0x2A,
    "Tab": 0x2B,
    "Space": 0x2C, " ": 0x2C,
    # Punctuation / symbols (US layout)
    "-": 0x2D, "=": 0x2E,
    "[": 0x2F, "]": 0x30,
    "\\": 0x31,
    ";": 0x33, "'": 0x34,
    "`": 0x35,
    ",": 0x36, ".": 0x37, "/": 0x38,
    # Lock keys
    "CapsLock": 0x39,
    # Function keys
    "F1": 0x3A, "F2": 0x3B, "F3": 0x3C, "F4": 0x3D,
    "F5": 0x3E, "F6": 0x3F, "F7": 0x40, "F8": 0x41,
    "F9": 0x42, "F10": 0x43, "F11": 0x44, "F12": 0x45,
    # Navigation
    "PrintScreen": 0x46,
    "ScrollLock": 0x47,
    "Pause": 0x48,
    "Insert": 0x49,
    "Home": 0x4A,
    "PageUp": 0x4B,
    "Delete": 0x4C,
    "End": 0x4D,
    "PageDown": 0x4E,
    "Right": 0x4F, "Left": 0x50,
    "Down": 0x51, "Up": 0x52,
}

# Characters that require Shift to type (US keyboard layout)
SHIFT_CHARS: dict[str, str] = {
    "!": "1", "@": "2", "#": "3", "$": "4",
    "%": "5", "^": "6", "&": "7", "*": "8",
    "(": "9", ")": "0", "_": "-", "+": "=",
    "{": "[", "}": "]", "|": "\\",
    ":": ";", '"': "'", "~": "`",
    "<": ",", ">": ".", "?": "/",
    "A": "a", "B": "b", "C": "c", "D": "d",
    "E": "e", "F": "f", "G": "g", "H": "h",
    "I": "i", "J": "j", "K": "k", "L": "l",
    "M": "m", "N": "n", "O": "o", "P": "p",
    "Q": "q", "R": "r", "S": "s", "T": "t",
    "U": "u", "V": "v", "W": "w", "X": "x",
    "Y": "y", "Z": "z",
}


def char_to_hid(char: str) -> tuple[int, int]:
    """Convert a single character to (modifier_byte, scan_code).

    Returns:
        Tuple of (modifier bitmask, HID scan code).

    Raises:
        ValueError: If the character has no known HID mapping.
    """
    if char in KEY_CODES:
        return (MODIFIER_NONE, KEY_CODES[char])
    if char in SHIFT_CHARS:
        base_char = SHIFT_CHARS[char]
        if base_char in KEY_CODES:
            return (MODIFIER_LEFT_SHIFT, KEY_CODES[base_char])
    raise ValueError(f"No HID mapping for character: {char!r}")


def key_name_to_hid(key: str) -> int:
    """Convert a key name to its HID scan code.

    An empty string returns ``0`` so callers can request a
    "modifier-only" tap (e.g. Super alone to open GNOME activities)
    by passing ``key=""`` to ``send_key_combo``.

    Raises:
        ValueError: If the key name is not recognized.
    """
    if key == "":
        return 0
    if key in KEY_CODES:
        return KEY_CODES[key]
    # Try case-insensitive lookup for single chars
    if len(key) == 1 and key.lower() in KEY_CODES:
        return KEY_CODES[key.lower()]
    raise ValueError(f"Unknown key name: {key!r}")


def modifiers_to_bitmask(modifiers: list[str]) -> int:
    """Convert a list of modifier names to a combined bitmask.

    Raises:
        ValueError: If any modifier name is not recognized.
    """
    bitmask = MODIFIER_NONE
    for mod in modifiers:
        mod_lower = mod.lower()
        if mod_lower not in MODIFIER_MAP:
            raise ValueError(f"Unknown modifier: {mod!r}")
        bitmask |= MODIFIER_MAP[mod_lower]
    return bitmask
