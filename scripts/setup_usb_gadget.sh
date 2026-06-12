#!/usr/bin/env bash
# setup_usb_gadget.sh — Configure Raspberry Pi Zero as a USB gadget.
#
# This script uses Linux USB ConfigFS to set up the Pi as a composite
# USB device. It supports three modes:
#   hid  — HID keyboard+mouse only (default, backward compatible)
#   ecm  — ECM (USB Ethernet) only (avoids "Keyboard Setup Assistant" on Mac)
#   all  — Both HID + ECM in one composite gadget
#
# Prerequisites:
#   - Raspberry Pi Zero (W/2W) or Pi 4 with USB OTG support
#   - dwc2 overlay enabled in /boot/config.txt
#   - libcomposite kernel module available
#
# Usage:
#   sudo bash scripts/setup_usb_gadget.sh [hid|ecm|all]           # set up
#   sudo bash scripts/setup_usb_gadget.sh teardown                  # tear down
#
# After setup:
#   hid mode: /dev/hidg0 (keyboard), /dev/hidg1 (mouse)
#   ecm mode: usb0 interface at 10.0.0.2/24
#   all mode: both of the above

set -euo pipefail

GADGET_DIR="/sys/kernel/config/usb_gadget/handsneyes_kb"
# Pre-rename deployments created the gadget under this name; a boot-
# time gadget with this name still owns the UDC on migrated Pis, so
# teardown must know about it or every (re)bind fails with EBUSY.
LEGACY_GADGET_DIR="/sys/kernel/config/usb_gadget/terminaleyes_kb"
UDC_PATH="/sys/class/udc"

# HID keyboard report descriptor (standard 8-byte boot keyboard)
# Modifier byte + reserved + 6 key codes
KEYBOARD_REPORT_DESCRIPTOR=$(printf '%b' \
    '\x05\x01' \
    '\x09\x06' \
    '\xa1\x01' \
    '\x05\x07' \
    '\x19\xe0' \
    '\x29\xe7' \
    '\x15\x00' \
    '\x25\x01' \
    '\x75\x01' \
    '\x95\x08' \
    '\x81\x02' \
    '\x95\x01' \
    '\x75\x08' \
    '\x81\x01' \
    '\x95\x06' \
    '\x75\x08' \
    '\x15\x00' \
    '\x25\x65' \
    '\x05\x07' \
    '\x19\x00' \
    '\x29\x65' \
    '\x81\x00' \
    '\xc0')

# HID mouse report descriptor (standard 4-byte boot mouse)
# Buttons (3) + X delta + Y delta + wheel
MOUSE_REPORT_DESCRIPTOR=$(printf '%b' \
    '\x05\x01' \
    '\x09\x02' \
    '\xa1\x01' \
    '\x09\x01' \
    '\xa1\x00' \
    '\x05\x09' \
    '\x19\x01' \
    '\x29\x03' \
    '\x15\x00' \
    '\x25\x01' \
    '\x95\x03' \
    '\x75\x01' \
    '\x81\x02' \
    '\x95\x01' \
    '\x75\x05' \
    '\x81\x01' \
    '\x05\x01' \
    '\x09\x30' \
    '\x09\x31' \
    '\x15\x81' \
    '\x25\x7f' \
    '\x75\x08' \
    '\x95\x02' \
    '\x81\x06' \
    '\x09\x38' \
    '\x15\x81' \
    '\x25\x7f' \
    '\x75\x08' \
    '\x95\x01' \
    '\x81\x06' \
    '\xc0' \
    '\xc0')

_teardown_one() {
    local dir="$1"
    if [ -d "$dir" ]; then
        # Disable the gadget
        echo "" > "$dir/UDC" 2>/dev/null || true
        # Remove functions from configuration
        rm -f "$dir/configs/c.1/ecm.usb0" 2>/dev/null || true
        rm -f "$dir/configs/c.1/hid.usb1" 2>/dev/null || true
        rm -f "$dir/configs/c.1/hid.usb0" 2>/dev/null || true
        # Remove strings
        rmdir "$dir/configs/c.1/strings/0x409" 2>/dev/null || true
        rmdir "$dir/configs/c.1" 2>/dev/null || true
        rmdir "$dir/functions/ecm.usb0" 2>/dev/null || true
        rmdir "$dir/functions/hid.usb1" 2>/dev/null || true
        rmdir "$dir/functions/hid.usb0" 2>/dev/null || true
        rmdir "$dir/strings/0x409" 2>/dev/null || true
        rmdir "$dir" 2>/dev/null || true
        echo "Gadget removed: $dir"
    fi
}

teardown() {
    echo "Tearing down USB gadget..."
    if [ ! -d "$GADGET_DIR" ] && [ ! -d "$LEGACY_GADGET_DIR" ]; then
        echo "No gadget found at $GADGET_DIR"
        return
    fi
    _teardown_one "$GADGET_DIR"
    _teardown_one "$LEGACY_GADGET_DIR"
}

setup() {
    local mode="${1:-hid}"

    # Ensure we're root
    if [ "$(id -u)" -ne 0 ]; then
        echo "ERROR: Must run as root (sudo)" >&2
        exit 1
    fi

    # Idempotent fast-path: if the requested setup is already live,
    # do NOT tear down and recreate — the ECM link carries the dev
    # mac's SSH + API traffic, and a teardown drops it mid-restart
    # (the service restarts on every crash, so this would flap the
    # network on every recovery).
    if [ "$mode" = "ecm" ] \
        && ip addr show usb0 2>/dev/null | grep -q "10\.0\.0\.2"; then
        bound=$(cat "$GADGET_DIR/UDC" 2>/dev/null \
            || cat "$LEGACY_GADGET_DIR/UDC" 2>/dev/null || true)
        if [ -n "$bound" ]; then
            echo "ECM gadget already live (UDC: $bound) — leaving as is."
            return 0
        fi
    fi

    # Check for existing gadget (either name)
    if [ -d "$GADGET_DIR" ] || [ -d "$LEGACY_GADGET_DIR" ]; then
        echo "Gadget already exists. Tearing down first..."
        teardown
    fi

    # Load the required kernel module (skip if already loaded)
    if ! lsmod | grep -q libcomposite; then
        /usr/sbin/modprobe libcomposite 2>/dev/null || modprobe libcomposite || {
            echo "ERROR: Cannot load libcomposite. Ensure dwc2 overlay is enabled:" >&2
            echo "  Add 'dtoverlay=dwc2' to /boot/config.txt" >&2
            echo "  Add 'dwc2' to /etc/modules" >&2
            echo "  Add 'libcomposite' to /etc/modules" >&2
            exit 1
        }
    fi

    # Determine product string based on mode
    local product_string
    case "$mode" in
        hid) product_string="Pi USB Keyboard+Mouse" ;;
        ecm) product_string="Pi USB Ethernet" ;;
        all) product_string="Pi USB Keyboard+Mouse+Ethernet" ;;
    esac

    echo "Creating USB gadget (mode: $mode)..."

    # Create the gadget
    mkdir -p "$GADGET_DIR"

    # USB device descriptor
    echo 0x1d6b > "$GADGET_DIR/idVendor"   # Linux Foundation
    echo 0x0104 > "$GADGET_DIR/idProduct"   # Multifunction Composite Gadget
    echo 0x0100 > "$GADGET_DIR/bcdDevice"   # v1.0.0
    echo 0x0200 > "$GADGET_DIR/bcdUSB"      # USB 2.0

    # Device strings
    mkdir -p "$GADGET_DIR/strings/0x409"
    echo "handsneyes0001"   > "$GADGET_DIR/strings/0x409/serialnumber"
    echo "handsneyes"       > "$GADGET_DIR/strings/0x409/manufacturer"
    echo "$product_string"    > "$GADGET_DIR/strings/0x409/product"

    # Configuration
    mkdir -p "$GADGET_DIR/configs/c.1/strings/0x409"
    echo "$product_string Configuration" > "$GADGET_DIR/configs/c.1/strings/0x409/configuration"
    echo 250 > "$GADGET_DIR/configs/c.1/MaxPower"  # 250mA

    # --- HID functions (hid or all mode) ---
    if [ "$mode" = "hid" ] || [ "$mode" = "all" ]; then
        # Keyboard HID function
        mkdir -p "$GADGET_DIR/functions/hid.usb0"
        echo 1   > "$GADGET_DIR/functions/hid.usb0/protocol"     # Keyboard
        echo 1   > "$GADGET_DIR/functions/hid.usb0/subclass"     # Boot interface
        echo 8   > "$GADGET_DIR/functions/hid.usb0/report_length"
        echo -ne "$KEYBOARD_REPORT_DESCRIPTOR" > "$GADGET_DIR/functions/hid.usb0/report_desc"

        # Link keyboard function to configuration
        ln -s "$GADGET_DIR/functions/hid.usb0" "$GADGET_DIR/configs/c.1/"

        # Mouse HID function
        mkdir -p "$GADGET_DIR/functions/hid.usb1"
        echo 2   > "$GADGET_DIR/functions/hid.usb1/protocol"     # Mouse
        echo 1   > "$GADGET_DIR/functions/hid.usb1/subclass"     # Boot interface
        echo 4   > "$GADGET_DIR/functions/hid.usb1/report_length"
        echo -ne "$MOUSE_REPORT_DESCRIPTOR" > "$GADGET_DIR/functions/hid.usb1/report_desc"

        # Link mouse function to configuration
        ln -s "$GADGET_DIR/functions/hid.usb1" "$GADGET_DIR/configs/c.1/"
    fi

    # --- ECM function (ecm or all mode) ---
    if [ "$mode" = "ecm" ] || [ "$mode" = "all" ]; then
        mkdir -p "$GADGET_DIR/functions/ecm.usb0"
        # Fixed MAC addresses — host (Mac) and device (Pi)
        echo "48:6f:73:74:00:01" > "$GADGET_DIR/functions/ecm.usb0/host_addr"
        echo "48:6f:73:74:00:02" > "$GADGET_DIR/functions/ecm.usb0/dev_addr"
        ln -s "$GADGET_DIR/functions/ecm.usb0" "$GADGET_DIR/configs/c.1/"
    fi

    # Enable the gadget by binding to the UDC (USB Device Controller)
    UDC=$(ls "$UDC_PATH" | head -1)
    if [ -z "$UDC" ]; then
        echo "ERROR: No USB Device Controller found. Is dwc2 loaded?" >&2
        teardown
        exit 1
    fi
    echo "$UDC" > "$GADGET_DIR/UDC"

    # --- Post-bind: configure ECM network interface ---
    if [ "$mode" = "ecm" ] || [ "$mode" = "all" ]; then
        echo "Waiting for usb0 interface..."
        sleep 1
        ip link set usb0 up
        ip addr add 10.0.0.2/24 dev usb0 2>/dev/null || true  # ignore if already set
        echo "  ECM Ethernet: usb0 at 10.0.0.2/24"
    fi

    echo "USB gadget enabled (mode: $mode)."
    if [ "$mode" = "hid" ] || [ "$mode" = "all" ]; then
        echo "  Keyboard: /dev/hidg0"
        echo "  Mouse:    /dev/hidg1"
    fi
    echo "  UDC:      $UDC"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-setup}" in
    teardown|remove|down)
        teardown
        ;;
    hid|ecm|all)
        setup "$1"
        ;;
    setup|up|"")
        setup "hid"
        ;;
    *)
        echo "Usage: $0 [hid|ecm|all|teardown]" >&2
        exit 1
        ;;
esac
