#!/usr/bin/env bash
# macos-devmouse-autoconnect.sh — keep the Pi BT HID device ("devmouse")
# connected to THIS Mac automatically, the way a real Bluetooth mouse
# reconnects on its own when it comes back into range.
#
# Why this exists: macOS will not let the Pi *initiate* the HID L2CAP
# channels (the "BR/EDR up but HID not" deadlock the Pi watchdog logs).
# Only the Mac can open them. The HIDNormallyConnectable SDP flag makes
# macOS willing to auto-initiate, but in practice macOS is lazy about it
# after sleep/wake or range loss. This agent closes that gap: it polls,
# and whenever devmouse is paired-but-disconnected it runs the Mac-
# initiated connect (which always works) — so reconnection is automatic
# and within POLL_SECONDS, no manual clicking in System Settings.
#
# Install (once):
#   scripts/macos-devmouse-autoconnect.sh --install
# Uninstall:
#   scripts/macos-devmouse-autoconnect.sh --uninstall
# Run the poll loop in the foreground (what the LaunchAgent invokes):
#   scripts/macos-devmouse-autoconnect.sh
#
# Requires: blueutil (brew install blueutil).

set -uo pipefail

# devmouse's Bluetooth MAC (the Pi adapter). blueutil wants dashes.
DEVMOUSE_MAC="${DEVMOUSE_MAC:-B8-27-EB-E7-2B-70}"
POLL_SECONDS="${POLL_SECONDS:-20}"
LABEL="com.handsneyes.devmouse-autoconnect"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

_blueutil() {
    if command -v blueutil >/dev/null 2>&1; then
        blueutil "$@"
    elif [ -x /opt/homebrew/bin/blueutil ]; then
        /opt/homebrew/bin/blueutil "$@"
    elif [ -x /usr/local/bin/blueutil ]; then
        /usr/local/bin/blueutil "$@"
    else
        echo "blueutil not found (brew install blueutil)" >&2
        return 127
    fi
}

is_acl_connected() {
    # BR/EDR ACL link only — NOT proof the HID profile is usable.
    [ "$(_blueutil --is-connected "$DEVMOUSE_MAC" 2>/dev/null)" = "1" ]
}

hid_active() {
    # The real test: macOS only publishes IOHIDInterface nodes for
    # devmouse (keyboard + mouse = 2) when the HID L2CAP channels are
    # actually open. After a passive/native reconnect macOS often
    # restores BR/EDR but leaves HID down — is-connected returns 1
    # while this returns 0. This is a Mac-ONLY signal: no IP path to
    # the Pi required, so it works on a remote target too.
    local n
    n="$(ioreg -l -w0 -r -c IOHIDInterface 2>/dev/null | grep -c devmouse)"
    [ "${n:-0}" -ge 1 ]
}

is_paired() {
    # blueutil has no --is-paired; --info succeeds only for a paired
    # device and prints "paired" in its output.
    _blueutil --info "$DEVMOUSE_MAC" 2>/dev/null | grep -q "paired"
}

reconnect() {
    # If BR/EDR is up but HID is down, a plain --connect is a no-op
    # (macOS thinks it's already connected). Tear the ACL link down
    # first so --connect re-runs the full HID profile open.
    if is_acl_connected; then
        _blueutil --disconnect "$DEVMOUSE_MAC" 2>/dev/null
        sleep 3
    fi
    # Retry: the first connect after wake/range-return frequently
    # returns Page Timeout before the radio settles.
    for _ in 1 2 3 4; do
        _blueutil --connect "$DEVMOUSE_MAC" 2>/dev/null
        sleep 4
        if hid_active; then
            return 0
        fi
    done
    hid_active
}

poll_loop() {
    echo "devmouse autoconnect: watching $DEVMOUSE_MAC every ${POLL_SECONDS}s"
    while true; do
        # Respect a deliberate "Forget This Device": only act while
        # still paired. Reconnect whenever HID is not actually up,
        # even if BR/EDR shows connected.
        if is_paired && ! hid_active; then
            echo "$(date '+%H:%M:%S') devmouse HID down — reconnecting"
            if reconnect; then
                echo "$(date '+%H:%M:%S') devmouse HID up"
            else
                echo "$(date '+%H:%M:%S') reconnect failed — will retry"
            fi
        fi
        sleep "$POLL_SECONDS"
    done
}

install_agent() {
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${SCRIPT_PATH}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DEVMOUSE_MAC</key>
        <string>${DEVMOUSE_MAC}</string>
        <key>POLL_SECONDS</key>
        <string>${POLL_SECONDS}</string>
    </dict>
    <!-- Keep the poll loop alive across crashes and logout/login. -->
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/devmouse-autoconnect.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/devmouse-autoconnect.log</string>
</dict>
</plist>
PLIST_EOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "installed + loaded: $PLIST"
    echo "log: /tmp/devmouse-autoconnect.log"
}

uninstall_agent() {
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "uninstalled: $PLIST"
}

case "${1:-}" in
    --install)   install_agent ;;
    --uninstall) uninstall_agent ;;
    *)           poll_loop ;;
esac
