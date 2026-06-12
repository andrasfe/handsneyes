#!/usr/bin/env bash
# radio_mode.sh — Switch Pi Zero 2 W between WiFi and Bluetooth modes.
#
# The BCM43436s has a shared radio — WiFi and BT Classic can't coexist
# reliably. This script toggles between modes and persists the choice.
#
# SSH over USB ECM (10.0.0.2) is always available regardless of mode.
#
# Usage:
#   sudo bash scripts/radio_mode.sh wifi    # Enable WiFi, disable BT
#   sudo bash scripts/radio_mode.sh bt      # Enable BT, disable WiFi
#   sudo bash scripts/radio_mode.sh apply   # Apply saved mode (for boot)
#   bash scripts/radio_mode.sh status       # Show current state

set -euo pipefail

# systemd units run with a minimal PATH that misses /usr/sbin —
# where rfkill lives. Without this, the bt_mode_live() check fails
# silently and `apply` cycles bluetoothd on EVERY service start.
PATH="/usr/sbin:/sbin:$PATH"

MODE_FILE="/home/andras/handsneyes/.radio-mode"
# BT is the product (the rig exists to serve BT HID); wifi is the
# maintenance mode. Defaulting to wifi once caused a silent post-
# reboot outage: boot applied wifi, the BT radio stayed down, and
# the target Mac's Connect clicks had nothing to talk to.
DEFAULT_MODE="bt"

get_saved_mode() {
    if [ -f "$MODE_FILE" ]; then
        cat "$MODE_FILE"
    else
        echo "$DEFAULT_MODE"
    fi
}

save_mode() {
    echo "$1" > "$MODE_FILE"
}

set_wifi_mode() {
    echo "Switching to WiFi mode (BT disabled)..."
    rfkill block bluetooth 2>/dev/null || true
    rfkill unblock wifi 2>/dev/null || true
    sleep 1
    nmcli radio wifi on 2>/dev/null || true
    nmcli connection up home-wifi 2>/dev/null || true
    save_mode "wifi"
    echo "  WiFi: enabled"
    echo "  Bluetooth: disabled"
}

set_bt_mode() {
    echo "Switching to Bluetooth mode (WiFi disabled)..."
    nmcli radio wifi off 2>/dev/null || true
    rfkill block wifi 2>/dev/null || true
    rfkill unblock bluetooth 2>/dev/null || true
    sleep 1
    systemctl restart bluetooth 2>/dev/null || true
    sleep 1
    # Power on adapter and make discoverable
    hciconfig hci0 up 2>/dev/null || true
    hciconfig hci0 class 0x0025C0 2>/dev/null || true
    hciconfig hci0 piscan 2>/dev/null || true
    save_mode "bt"
    echo "  WiFi: disabled"
    echo "  Bluetooth: enabled + discoverable"
}

bt_mode_live() {
    rfkill list bluetooth 2>/dev/null | grep -q "Soft blocked: no" \
        && hciconfig hci0 2>/dev/null | grep -q "UP RUNNING"
}

wifi_mode_live() {
    rfkill list wifi 2>/dev/null | grep -q "Soft blocked: no" \
        && ! hciconfig hci0 2>/dev/null | grep -q "UP RUNNING"
}

show_status() {
    local saved_mode
    saved_mode=$(get_saved_mode)
    echo "Saved mode: $saved_mode"
    echo ""
    echo "rfkill state:"
    rfkill list 2>/dev/null || echo "  (rfkill not available)"
    echo ""
    echo "WiFi:"
    nmcli -t -f DEVICE,STATE device status 2>/dev/null | grep wlan0 || echo "  not available"
    echo ""
    echo "Bluetooth:"
    hciconfig hci0 2>/dev/null || echo "  not available"
}

case "${1:-status}" in
    wifi)
        set_wifi_mode
        echo ""
        echo "Reboot to fully apply, or mode is active now."
        ;;
    bt)
        set_bt_mode
        echo ""
        echo "Reboot to fully apply, or mode is active now."
        ;;
    apply)
        # Boot/service-start path — must be IDEMPOTENT. set_bt_mode
        # restarts bluetoothd, and bluetoothd restarts trigger a
        # gateway restart (bluetooth.service.d/restart-hid-gateway
        # drop-in re-registers the HID profile). If apply cycled
        # bluetooth unconditionally, every gateway start would kill
        # itself through that hook.
        mode=$(get_saved_mode)
        echo "Applying saved radio mode: $mode"
        case "$mode" in
            bt)
                if bt_mode_live; then
                    # Cheap non-disruptive re-asserts only.
                    hciconfig hci0 piscan 2>/dev/null || true
                    echo "BT mode already live — radios left as is."
                else
                    set_bt_mode
                fi
                ;;
            *)
                if wifi_mode_live; then
                    echo "WiFi mode already live — radios left as is."
                else
                    set_wifi_mode
                fi
                ;;
        esac
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [wifi|bt|apply|status]" >&2
        exit 1
        ;;
esac
