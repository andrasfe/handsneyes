#!/usr/bin/env bash
# setup_bt_hid.sh — Configure Raspberry Pi as a Bluetooth HID keyboard + mouse.
#
# Installs BlueZ, configures the adapter for HID, and prepares the Pi
# to act as a Bluetooth combo device (keyboard + mouse) that can be
# paired with a host.
#
# CRITICAL: Disables BlueZ's `input` plugin so our L2CAP sockets on
# PSM 17/19 aren't stolen.  This is required by ALL BT HID emulator
# projects (PiKVM, Bluetooth_HID, Pi-Bluetooth-Keyboard, etc.).
#
# Prerequisites:
#   - Raspberry Pi with built-in Bluetooth (Pi Zero 2 W, Pi 3/4/5)
#   - handsneyes installed with: pip install -e ".[rpi]"
#
# Usage:
#   sudo bash scripts/setup_bt_hid.sh          # set up
#   sudo bash scripts/setup_bt_hid.sh teardown  # remove configuration

set -euo pipefail

CONF_FILE="/etc/bluetooth/main.conf"
CONF_BACKUP="/etc/bluetooth/main.conf.bak"
OVERRIDE_DIR="/etc/systemd/system/bluetooth.service.d"
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"
AGENT_SCRIPT="/usr/local/bin/bt-pairing-agent.sh"
AGENT_CONF="/etc/systemd/system/bt-agent.service"

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
teardown() {
    echo "Tearing down Bluetooth HID configuration..."

    # Restore original bluetooth config
    if [ -f "$CONF_BACKUP" ]; then
        cp "$CONF_BACKUP" "$CONF_FILE"
        echo "Restored original $CONF_FILE"
    fi

    # Remove bluetoothd override
    if [ -f "$OVERRIDE_FILE" ]; then
        rm -f "$OVERRIDE_FILE"
        rmdir "$OVERRIDE_DIR" 2>/dev/null || true
        echo "Removed bluetoothd override"
    fi

    # Remove the auto-accept agent service
    if [ -f "$AGENT_CONF" ]; then
        systemctl stop bt-agent 2>/dev/null || true
        systemctl disable bt-agent 2>/dev/null || true
        rm -f "$AGENT_CONF"
        rm -f "$AGENT_SCRIPT"
        systemctl daemon-reload
        echo "Removed bt-agent service"
    fi

    # Remove the audio-SDP stripper
    if [ -f "/etc/systemd/system/bt-strip-audio-sdp.service" ]; then
        systemctl disable bt-strip-audio-sdp 2>/dev/null || true
        rm -f /etc/systemd/system/bt-strip-audio-sdp.service
        rm -f /usr/local/bin/bt-strip-audio-sdp.sh
        systemctl daemon-reload
        echo "Removed bt-strip-audio-sdp service"
    fi

    # Restart bluetooth
    systemctl daemon-reload
    systemctl restart bluetooth
    echo "Bluetooth HID configuration removed."
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup() {
    # Ensure we're root
    if [ "$(id -u)" -ne 0 ]; then
        echo "ERROR: Must run as root (sudo)" >&2
        exit 1
    fi

    echo "=== Setting up Bluetooth HID (Keyboard + Mouse) ==="
    echo ""

    # -----------------------------------------------------------------------
    # Step 1: Install dependencies
    # -----------------------------------------------------------------------
    echo "[1/6] Installing Bluetooth dependencies..."

    apt-get update -qq
    apt-get install -y -qq bluez python3-dbus bluetooth pi-bluetooth

    echo "  BlueZ version: $(bluetoothctl --version 2>/dev/null || echo 'unknown')"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 2: Configure bluetoothd with --compat and --noplugin=input,a2dp,...
    # -----------------------------------------------------------------------
    echo "[2/6] Configuring bluetoothd (disable input + audio plugins)..."

    # CRITICAL: The BlueZ 'input' plugin binds to PSM 17/19 (HID control
    # and interrupt channels).  If loaded, our application gets EADDRINUSE
    # when trying to bind its own L2CAP sockets.  Every working BT HID
    # project requires disabling it.
    #
    # ALSO disabled: every audio-side plugin (a2dp-sink, a2dp-source,
    # avrcp, hfp, hsp, gateway, media).  Without this, the moment macOS
    # pairs with the Pi it ALSO offers it as a Bluetooth audio output
    # device — and routes the Mac's system sound to it.  Operator then
    # loses Mac speaker audio every time the Pi is connected.  The Pi
    # advertises nothing useful audio-wise (no DAC, no speaker driver
    # in this build), so the right answer is to stop bluetoothd from
    # ever loading the plugins in the first place.  Once they're gone
    # from the SDP record macOS no longer treats the Pi as audio.
    #
    # --compat enables the SDP server socket so sdptool and
    # RegisterProfile SDP records are visible to remote devices.

    mkdir -p "$OVERRIDE_DIR"

    # Find the actual bluetoothd binary path
    BTDAEMON="/usr/libexec/bluetooth/bluetoothd"
    if [ ! -f "$BTDAEMON" ]; then
        BTDAEMON="/usr/lib/bluetooth/bluetoothd"
    fi

    cat > "$OVERRIDE_FILE" <<EOF
[Service]
ExecStart=
ExecStart=$BTDAEMON --compat --noplugin=input,a2dp,avrcp,hfp,hsp,gateway,media,audio
EOF

    echo "  Override: $OVERRIDE_FILE"
    echo "  Flags: --compat --noplugin=input,a2dp,avrcp,hfp,hsp,gateway,media,audio"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 3: Configure Bluetooth adapter
    # -----------------------------------------------------------------------
    echo "[3/6] Configuring Bluetooth adapter..."

    # Backup original config
    if [ -f "$CONF_FILE" ] && [ ! -f "$CONF_BACKUP" ]; then
        cp "$CONF_FILE" "$CONF_BACKUP"
    fi

    # Device class 0x0025C0:
    #   Major class: Peripheral (0x0500)
    #   Minor class: Combo keyboard+pointing (0x00C0)
    #   Service class: none extra
    cat > "$CONF_FILE" <<'BTCONF'
[General]
# devmouse — Bluetooth HID (Keyboard + Mouse)
Name = devmouse
Class = 0x0025C0
# BR/EDR-only. The HID gateway is a classic-Bluetooth device and needs
# no LE. Critically, a paired Mac keeps opening an Apple-Continuity LE
# link to this adapter; on the Pi Zero 2 W's combo radio that LE link
# wedges in "state 5" and jams the whole controller (every HCI op then
# returns "Connection timed out"), which blocks the BR/EDR HID
# reconnect — the recurring "can't connect the remote Mac" failure.
# Forcing bredr mode means the Mac can't open the LE link at all.
ControllerMode = bredr
DiscoverableTimeout = 0
PairableTimeout = 0
Discoverable = true
# FastConnectable keeps the adapter in fast page-scan mode so the
# host (Mac) re-pages and reconnects within a second or two when it
# comes back into range, instead of after the default slow-scan
# interval. Pairs with the SDP HIDNormallyConnectable flag (bt_hid.py)
# to make devmouse auto-reconnect like a real BT mouse.
FastConnectable = true
# Pairable is NOT a valid main.conf key in BlueZ 5.82 — bluetoothd
# warns "Unknown key Pairable for group General" and ignores it.
# The correct path is bluetoothctl `pairable on` at runtime, which
# bt-strip-audio-sdp.sh re-asserts after every bluetoothd cycle.

[Policy]
AutoEnable=true
BTCONF

    echo "  Set device class to 0x0025C0 (Keyboard + Mouse combo)"
    echo "  Discoverable: always, no timeout"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 4: Set up auto-accept pairing agent (Python D-Bus agent)
    # -----------------------------------------------------------------------
    echo "[4/6] Setting up Python D-Bus pairing agent..."

    # bluetoothctl-based agents (the heredoc/pipe trick) routinely
    # fail with "Failed to register agent object" on recent BlueZ —
    # the interactive bluetoothctl process exits before the D-Bus
    # session that owns the agent finishes registering it.  The
    # Python agent registers via dbus directly and stays in a GLib
    # main loop, which is the pattern every working BT HID project
    # converges on (see CLAUDE.md "BT HID Architecture").
    AGENT_PY_DEST="/usr/local/bin/bt-agent.py"
    REPO_AGENT_PY="$(dirname "$(readlink -f "$0")")/bt-agent.py"
    if [ -f "$REPO_AGENT_PY" ]; then
        cp "$REPO_AGENT_PY" "$AGENT_PY_DEST"
        chmod +x "$AGENT_PY_DEST"
        # Make sure dbus python binding is installed (apt install in
        # step [1/6] already does this, but be defensive).
        apt-get install -y -qq python3-dbus python3-gi >/dev/null 2>&1 || true
    else
        echo "  ERROR: $REPO_AGENT_PY not found — cannot install agent."
        echo "  Pairing will fail.  Copy scripts/bt-agent.py to the Pi"
        echo "  and re-run this script."
        return 1
    fi

    cat > "$AGENT_CONF" <<AGENTSERVICE
[Unit]
Description=Bluetooth Auto-Accept Pairing Agent (D-Bus, Python)
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStartPre=/bin/sleep 3
ExecStart=/usr/bin/python3 $AGENT_PY_DEST
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
AGENTSERVICE

    # Tidy up the previous bash-shim if it was installed by an
    # earlier version of this script.
    rm -f /usr/local/bin/bt-pairing-agent.sh

    systemctl daemon-reload
    systemctl enable bt-agent
    echo "  Pairing agent: $AGENT_PY_DEST"
    echo "  Auto-accept: NoInputNoOutput (Just Works pairing)"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 4.25: Remove stale Pi-side pairings.
    # -----------------------------------------------------------------------
    # The Pi remembers every device it has ever paired with under
    # /var/lib/bluetooth/<adapter>/<peer-mac>/.  If the operator has
    # Forgotten the Pi on the Mac side without also removing it on the
    # Pi, the next pair attempt arrives with new credentials but the
    # Pi auto-rejects because it already has a (stale) bond — silently,
    # at HCI level, before the agent ever fires.  Symptom: Mac scan
    # finds the Pi but Pair just spins and times out, while
    # journalctl -u bt-agent shows nothing because the agent isn't
    # even being consulted.
    #
    # Wiping these as part of setup gives every fresh install a clean
    # pair surface and is harmless on a Pi that's never been paired.
    echo "[4.25/6] Removing stale Pi-side pairings..."
    STALE_DEVS=$(bluetoothctl devices 2>/dev/null | awk '/^Device / {print $2}')
    if [ -n "$STALE_DEVS" ]; then
        for MAC in $STALE_DEVS; do
            bluetoothctl remove "$MAC" >/dev/null 2>&1 || true
            echo "  removed $MAC"
        done
    else
        echo "  (none to remove)"
    fi
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 4.5: Install audio-SDP stripper
    # -----------------------------------------------------------------------
    echo "[4.5/6] Installing audio-SDP stripper..."

    # Even with --noplugin=a2dp,avrcp,hfp,hsp,gateway,media,audio,
    # bluetoothd still publishes Hands-Free / SIM Access SDP records
    # as part of its protocol stack — those flags are at plugin level
    # not at protocol level.  macOS sees those records on pair and
    # auto-routes the Mac's system sound to the Pi.  We strip them
    # after every bluetoothd start with this oneshot unit.
    STRIP_SCRIPT="/usr/local/bin/bt-strip-audio-sdp.sh"
    STRIP_UNIT="/etc/systemd/system/bt-strip-audio-sdp.service"
    REPO_SCRIPT="$(dirname "$(readlink -f "$0")")/bt-strip-audio-sdp.sh"

    if [ -f "$REPO_SCRIPT" ]; then
        cp "$REPO_SCRIPT" "$STRIP_SCRIPT"
        chmod +x "$STRIP_SCRIPT"

        cat > "$STRIP_UNIT" <<STRIPSERVICE
[Unit]
Description=Strip non-HID audio/telephony SDP records after bluetoothd starts
# PartOf= is what makes this re-fire whenever bluetooth.service
# restarts — without it the oneshot stays "active (exited)" forever
# and the HFP/SAP records come back the next time bluetoothd cycles
# (e.g. via radio_mode.sh apply during handsneyes-pi startup).
After=bluetooth.service handsneyes-pi.service
Requires=bluetooth.service
PartOf=bluetooth.service

[Service]
Type=oneshot
# Wait long enough for handsneyes-pi to call RegisterProfile +
# bluetoothd to publish the HID SDP record before we start tearing
# records down.  10s is conservative; pi-zero-2-w usually settles in
# 3-5s but the race is annoying when it happens.
ExecStartPre=/bin/sleep 10
ExecStart=$STRIP_SCRIPT

[Install]
WantedBy=bluetooth.service
STRIPSERVICE

        systemctl daemon-reload
        systemctl enable bt-strip-audio-sdp 2>/dev/null || true
        echo "  Stripper installed: $STRIP_SCRIPT"
        echo "  Unit: $STRIP_UNIT (oneshot, fires after bluetoothd)"
        echo "  OK"
    else
        echo "  WARNING: $REPO_SCRIPT not found — Pi will continue"
        echo "  advertising HFP/SAP SDP records (Mac will route audio)."
    fi

    # -----------------------------------------------------------------------
    # Step 5: Apply configuration and power on adapter
    # -----------------------------------------------------------------------
    echo "[5/6] Applying configuration..."

    # Unblock rfkill (common on Pi — BT starts soft-blocked)
    rfkill unblock bluetooth 2>/dev/null || true
    sleep 1

    # Restart bluetooth to pick up new config + override
    systemctl daemon-reload
    systemctl restart bluetooth
    sleep 2

    # Start the pairing agent
    systemctl restart bt-agent 2>/dev/null || true
    sleep 1

    # Power on via hciconfig (more reliable than bluetoothctl for class/name)
    hciconfig hci0 up 2>/dev/null || true
    hciconfig hci0 class 0x0025C0 2>/dev/null || true
    hciconfig hci0 name "devmouse" 2>/dev/null || true
    hciconfig hci0 piscan 2>/dev/null || true
    sleep 1

    echo "  Adapter powered on and discoverable"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 6: Verify
    # -----------------------------------------------------------------------
    echo "[6/6] Verifying setup..."

    # Check bluetoothd flags
    BTPID=$(pgrep -x bluetoothd 2>/dev/null || echo "")
    if [ -n "$BTPID" ]; then
        BTCMDLINE=$(cat "/proc/$BTPID/cmdline" 2>/dev/null | tr '\0' ' ' || echo "")
        if echo "$BTCMDLINE" | grep -qE "noplugin=[^ ]*input"; then
            echo "  bluetoothd: input plugin disabled (correct)"
        else
            echo "  WARNING: input plugin may still be loaded!"
            echo "  cmdline: $BTCMDLINE"
        fi
        if echo "$BTCMDLINE" | grep -qE "noplugin=[^ ]*a2dp"; then
            echo "  bluetoothd: audio plugins (a2dp/avrcp/hfp/hsp) disabled"
            echo "                — Pi will not appear as a Mac audio output"
        else
            echo "  WARNING: audio plugins may still be loaded — Mac will"
            echo "  route sound to the Pi when paired. Re-run this script."
        fi
        if echo "$BTCMDLINE" | grep -q "compat"; then
            echo "  bluetoothd: --compat mode (SDP server enabled)"
        fi
    fi

    # Check agent
    if systemctl is-active --quiet bt-agent 2>/dev/null; then
        echo "  bt-agent: running"
    else
        echo "  WARNING: bt-agent not running"
    fi

    # Check adapter status
    ADAPTER_INFO=$(bluetoothctl show 2>/dev/null || echo "")
    if echo "$ADAPTER_INFO" | grep -q "Powered: yes"; then
        echo "  Adapter: powered on"
    else
        echo "  WARNING: Adapter may not be powered on"
    fi

    if echo "$ADAPTER_INFO" | grep -q "Discoverable: yes"; then
        echo "  Discoverable: yes"
    else
        echo "  WARNING: Adapter may not be discoverable"
    fi

    BT_ADDR=$(echo "$ADAPTER_INFO" | grep "Controller" | awk '{print $2}')
    if [ -n "$BT_ADDR" ]; then
        echo "  Address: $BT_ADDR"
    fi

    echo ""
    echo "=== Bluetooth HID Setup Complete ==="
    echo ""
    echo "The Pi is now discoverable as 'devmouse'."
    echo "It presents as a combo keyboard + mouse device."
    echo ""
    echo "Next steps:"
    echo "  1. Start the REST API: sudo systemctl start handsneyes-pi"
    echo "  2. On the host machine, pair with 'devmouse'"
    echo "  3. The REST API accepts Bluetooth commands at:"
    echo ""
    echo "  Keyboard:"
    echo "     POST /bt/keystroke  {\"key\": \"Enter\"}"
    echo "     POST /bt/key-combo  {\"modifiers\": [\"ctrl\"], \"key\": \"c\"}"
    echo "     POST /bt/text       {\"text\": \"hello\"}"
    echo ""
    echo "  Mouse:"
    echo "     POST /bt/mouse/move   {\"x\": 10, \"y\": 0}"
    echo "     POST /bt/mouse/click  {\"button\": \"left\"}"
    echo "     POST /bt/mouse/scroll {\"amount\": -3}"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-setup}" in
    teardown|remove|down)
        teardown
        ;;
    setup|up|"")
        setup
        ;;
    *)
        echo "Usage: $0 [setup|teardown]" >&2
        exit 1
        ;;
esac
