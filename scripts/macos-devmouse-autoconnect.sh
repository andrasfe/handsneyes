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
# Pause auto-reconnect so you can disconnect manually and have it
# STAY disconnected (indefinite, or for N minutes), then resume:
#   scripts/macos-devmouse-autoconnect.sh --pause [N]
#   scripts/macos-devmouse-autoconnect.sh --resume
# Show current state:
#   scripts/macos-devmouse-autoconnect.sh --status
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
# While this file exists the agent will NOT reconnect — so you can
# disconnect devmouse from the menu bar and have it STAY disconnected
# (e.g. to drive a different machine, or just pause it). Created by
# `--pause`, removed by `--resume`. `--pause N` auto-resumes after N
# minutes.
PAUSE_FILE="${PAUSE_FILE:-$HOME/.config/handsneyes/devmouse-pause}"

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

paused() {
    # Honour a timed pause: if the pause file holds a future epoch,
    # stay paused until then, otherwise (empty / indefinite) stay
    # paused until --resume.
    [ -f "$PAUSE_FILE" ] || return 1
    local until
    until="$(cat "$PAUSE_FILE" 2>/dev/null)"
    if [ -n "$until" ] && [ "$until" -gt 0 ] 2>/dev/null; then
        if [ "$(date +%s)" -ge "$until" ]; then
            rm -f "$PAUSE_FILE"
            echo "$(date '+%H:%M:%S') pause expired — resuming"
            return 1
        fi
    fi
    return 0
}

poll_loop() {
    echo "devmouse autoconnect: watching $DEVMOUSE_MAC every ${POLL_SECONDS}s"
    while true; do
        # Respect a deliberate "Forget This Device" (only act while
        # still paired) and a manual pause (so the operator can
        # disconnect and have it stay disconnected). Otherwise
        # reconnect whenever HID is not actually up, even if BR/EDR
        # shows connected.
        if is_paired && ! paused && ! hid_active; then
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

pause_agent() {
    mkdir -p "$(dirname "$PAUSE_FILE")"
    local mins="${1:-}"
    if [ -n "$mins" ] && [ "$mins" -gt 0 ] 2>/dev/null; then
        echo "$(( $(date +%s) + mins * 60 ))" > "$PAUSE_FILE"
        echo "auto-reconnect paused for ${mins} min — you can disconnect devmouse now"
    else
        echo "0" > "$PAUSE_FILE"
        echo "auto-reconnect paused (indefinite) — resume with --resume"
    fi
    # Disconnect right away so the operator doesn't have to.
    _blueutil --disconnect "$DEVMOUSE_MAC" 2>/dev/null || true
}

resume_agent() {
    rm -f "$PAUSE_FILE"
    echo "auto-reconnect resumed"
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
    --pause)     pause_agent "${2:-}" ;;
    --resume)    resume_agent ;;
    --status)
        echo "paired:    $(is_paired && echo yes || echo no)"
        echo "hid_up:    $(hid_active && echo yes || echo no)"
        echo "paused:    $([ -f "$PAUSE_FILE" ] && echo yes || echo no)"
        echo "agent:     $(launchctl list 2>/dev/null | grep -q "$LABEL" && echo loaded || echo not-loaded)"
        ;;
    *)           poll_loop ;;
esac
