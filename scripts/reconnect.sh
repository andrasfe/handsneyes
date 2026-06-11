#!/usr/bin/env bash
# reconnect.sh — Restore USB ECM + BT HID connectivity after cable change or reboot.
# Run from the dev Mac. No arguments needed. Guides you through every step.
#
# Robustness rules this script follows:
#   - The :8080/health endpoint is the source of truth for "gateway
#     up" — NOT the systemd unit name. The Pi may run the gateway as
#     handsneyes-pi (migrated deploy) or terminaleyes-pi (original
#     deploy); both are the same code. The unit name is detected,
#     never assumed.
#   - No `set -e`: a failing probe is expected (that's why we're
#     reconnecting) and must be retried/reported, never allowed to
#     silently kill the script mid-step.
#   - Pi-side services are managed through systemd only (the units
#     carry Restart=always) — no ad-hoc setsid/nohup spawns that
#     systemd can't supervise.
#   - sudo on the Pi is passwordless for $PI_USER (`sudo -n`). No
#     passwords are embedded here.
#
# Env knobs:
#   RECONNECT_SKIP_BT=1   stop after the gateway API check (useful
#                         for scripted/smoke runs; BT needs operator
#                         action on the target Mac anyway).

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "  ${BLUE}[..]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[!!]${NC} $1"; }

wait_for_enter() {
    echo ""
    read -rp "  Press Enter when done..." _
    echo ""
}

PI_IP="10.0.0.2"
MAC_IP="10.0.0.1"
PI_USER="andras"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5"
# Gateway unit name on the Pi — detected in step 3.
PI_UNIT=""

pi_run() {  # pi_run <command...>  — run on the Pi, never aborts the script
    $SSH "$PI_USER@$PI_IP" "$@" 2>/dev/null
}

api_up() {
    curl -s --connect-timeout 2 "http://$PI_IP:8080/health" > /dev/null 2>&1
}

check_bt() {
    local health bt
    health=$(curl -s --connect-timeout 2 "http://$PI_IP:8080/health" 2>/dev/null || echo "{}")
    bt=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('bt_hid_connected',False))" 2>/dev/null || echo "False")
    [ "$bt" = "True" ]
}

detect_unit() {
    # Sets PI_UNIT to whichever gateway unit exists on the Pi.
    local u
    for u in handsneyes-pi terminaleyes-pi; do
        if pi_run "systemctl cat $u.service > /dev/null 2>&1 && echo yes" | grep -q yes; then
            PI_UNIT="$u"
            return 0
        fi
    done
    return 1
}

restart_gateway() {
    [ -n "$PI_UNIT" ] || detect_unit || return 1
    pi_run "sudo -n systemctl restart $PI_UNIT" || return 1
}

restart_bt_agent() {
    # bt-agent runs as a systemd unit with Restart=always; restarting
    # through systemd keeps it supervised (an ad-hoc spawn would die
    # unnoticed on the next crash).
    pi_run "sudo -n systemctl restart bt-agent" || true
}

# ===========================================================================
# Step 1: Find and configure USB Ethernet interface
# ===========================================================================
echo -e "\n${BLUE}=== Step 1: USB Ethernet ===${NC}"

find_iface() {
    # The Pi's ECM gadget always presents MAC 48:6f:73:74:xx:xx
    # ("host"). Plain for-loop — a pipeline-into-while here once
    # aborted the whole script under `set -e` when the last probed
    # interface didn't match.
    local i
    for i in $(ifconfig -l 2>/dev/null); do
        if ifconfig "$i" 2>/dev/null | grep -q "48:6f:73:74"; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

ATTEMPT=0
while true; do
    IFACE=$(find_iface || true)
    if [ -n "$IFACE" ]; then
        ok "Found interface: $IFACE"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -le 3 ]; then
        warn "No USB ECM interface found"
        echo "  Make sure the USB cable is connected between this Mac and the Pi."
        echo "  Try unplugging and replugging the cable."
        wait_for_enter
    else
        fail "Still no USB ECM interface after $ATTEMPT attempts."
        echo "  Possible causes:"
        echo "    - Wrong cable (must be data cable, not charge-only)"
        echo "    - Pi not powered on"
        echo "    - Pi USB gadget not configured (needs setup_usb_gadget.sh ecm)"
        echo "  Try power cycling the Pi: unplug power, wait 5s, replug."
        wait_for_enter
    fi
done

# Configure IP
CURRENT_IP=$(ifconfig "$IFACE" 2>/dev/null | grep "inet " | awk '{print $2}' || true)
if [ "$CURRENT_IP" = "$MAC_IP" ]; then
    ok "IP already set: $MAC_IP"
else
    info "Setting $IFACE to $MAC_IP ..."
    if sudo ifconfig "$IFACE" "$MAC_IP" netmask 255.255.255.0 up; then
        sleep 1
        ok "IP configured: $MAC_IP on $IFACE"
    else
        fail "Could not configure $IFACE — check sudo rights and rerun."
        exit 1
    fi
fi

# ===========================================================================
# Step 2: Wait for Pi
# ===========================================================================
echo -e "\n${BLUE}=== Step 2: Pi connectivity ===${NC}"

ATTEMPT=0
while true; do
    info "Pinging $PI_IP ..."
    TRIES=0
    while ! ping -c 1 -W 2 "$PI_IP" > /dev/null 2>&1; do
        TRIES=$((TRIES + 1))
        if [ "$TRIES" -ge 10 ]; then
            break
        fi
        sleep 2
    done

    if ping -c 1 -W 2 "$PI_IP" > /dev/null 2>&1; then
        ok "Pi reachable at $PI_IP"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -le 2 ]; then
        warn "Pi not responding at $PI_IP"
        echo "  The Pi may still be booting (takes ~30s)."
        echo "  If it's been more than a minute, try power cycling the Pi."
        wait_for_enter
    else
        warn "Still can't reach Pi"
        echo "  Try: unplug Pi power, wait 5s, replug, wait 30s."
        echo "  Then re-run this script."
        wait_for_enter
    fi
done

# ===========================================================================
# Step 3: HID gateway API (health-first)
# ===========================================================================
echo -e "\n${BLUE}=== Step 3: HID gateway ===${NC}"

if api_up; then
    ok "Gateway API responding at http://$PI_IP:8080"
    detect_unit || true
    [ -n "$PI_UNIT" ] && ok "Gateway unit: $PI_UNIT"
else
    info "API not responding — locating gateway unit ..."
    if ! detect_unit; then
        fail "No gateway unit found on the Pi (tried handsneyes-pi, terminaleyes-pi)."
        echo "  The Pi-side service was never installed (or SSH/sudo is broken)."
        echo "  Inspect: $SSH $PI_USER@$PI_IP 'systemctl list-units --type=service'"
        exit 1
    fi
    info "Starting $PI_UNIT ..."
    restart_gateway || warn "systemctl restart $PI_UNIT failed — will keep polling the API anyway"

    info "Waiting for API ..."
    TRIES=0
    while ! api_up; do
        TRIES=$((TRIES + 1))
        if [ "$TRIES" -ge 15 ]; then
            fail "API not responding after 30s"
            echo "  Logs: $SSH $PI_USER@$PI_IP 'sudo -n journalctl -u $PI_UNIT -n 50 --no-pager'"
            exit 1
        fi
        sleep 2
    done
    ok "Gateway API responding"
fi

# Pairing agent (systemd unit bt-agent, Restart=always)
if pi_run "pgrep -f bt-agent.py" > /dev/null; then
    ok "Pairing agent running"
else
    info "Starting pairing agent (systemctl restart bt-agent) ..."
    restart_bt_agent
    sleep 3
    if pi_run "pgrep -f bt-agent.py" > /dev/null; then
        ok "Pairing agent started"
    else
        warn "Pairing agent not running — pairing may hang"
        echo "  Logs: $SSH $PI_USER@$PI_IP 'sudo -n journalctl -u bt-agent -n 30 --no-pager'"
    fi
fi

if [ "${RECONNECT_SKIP_BT:-}" = "1" ]; then
    echo ""
    ok "RECONNECT_SKIP_BT=1 — stopping after gateway check."
    exit 0
fi

# ===========================================================================
# Step 4: Bluetooth HID connection
# ===========================================================================
echo -e "\n${BLUE}=== Step 4: Bluetooth HID ===${NC}"

if check_bt; then
    ok "BT HID already connected"
else
    ATTEMPT=0
    while true; do
        ATTEMPT=$((ATTEMPT + 1))

        if [ "$ATTEMPT" -le 1 ]; then
            warn "BT HID not connected"
            echo ""
            echo "  macOS will NOT auto-open the HID channel from the Pi side —"
            echo "  the target Mac must initiate. On the TARGET Mac:"
            echo "    1. Open System Settings → Bluetooth"
            echo "    2. Look for 'devmouse' or 'keyboarder' in Nearby Devices"
            echo "    3. Click Connect"
            echo "    4. Dismiss 'Keyboard Setup Assistant' if it appears"
        elif [ "$ATTEMPT" -le 2 ]; then
            warn "Still not connected"
            echo ""
            echo "  Try on the TARGET Mac:"
            echo "    1. If device shows 'Connected' but this script doesn't see it,"
            echo "       click Forget This Device, then reconnect"
            echo "    2. If device doesn't appear, toggle Bluetooth off and on"
        elif [ "$ATTEMPT" -le 3 ]; then
            warn "Still not connected — restarting BT stack on Pi"
            echo ""
            pi_run "sudo -n bash -c 'systemctl restart bluetooth; sleep 2; hciconfig hci0 up; hciconfig hci0 class 0x0025C0; hciconfig hci0 piscan'" || true
            restart_gateway || true
            sleep 5
            restart_bt_agent
            sleep 3
            echo "  Bluetooth restarted on Pi."
            echo "  On the TARGET Mac:"
            echo "    1. Forget 'devmouse' / 'keyboarder' if listed"
            echo "    2. Toggle Bluetooth off and on"
            echo "    3. Wait for device to appear, then Connect"
        else
            warn "Still not connected — clean slate"
            echo ""
            echo "  Removing all pairings on Pi and doing full reset..."
            pi_run "sudo -n bash -c '
                for dev in \$(bluetoothctl devices 2>/dev/null | awk \"{print \\\$2}\"); do
                    bluetoothctl remove \"\$dev\" > /dev/null 2>&1
                done
                systemctl restart bluetooth
                sleep 2
                hciconfig hci0 up
                hciconfig hci0 class 0x0025C0
                hciconfig hci0 piscan
            '" || true
            restart_gateway || true
            sleep 5
            restart_bt_agent
            sleep 3
            echo "  Full reset done."
            echo "  On the TARGET Mac:"
            echo "    1. Forget 'devmouse' / 'keyboarder' if listed"
            echo "    2. Toggle Bluetooth off and on"
            echo "    3. Wait for device to appear in Nearby Devices"
            echo "    4. Click Connect"
        fi

        echo ""
        echo "  Waiting for BT connection (polling every 2s) ..."
        POLL=0
        while ! check_bt; do
            POLL=$((POLL + 1))
            if [ "$POLL" -ge 30 ]; then
                break
            fi
            sleep 2
        done

        if check_bt; then
            ok "BT HID connected"
            break
        fi
    done
fi

# ===========================================================================
# Step 5: Verify everything works
# ===========================================================================
echo -e "\n${BLUE}=== Step 5: Verification ===${NC}"

# Test keyboard
RESULT=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
    -d '{"key":"a"}' "http://$PI_IP:8080/bt/keystroke" 2>/dev/null || echo "000")
if [ "$RESULT" = "200" ]; then
    ok "BT keyboard: working (sent 'a' to target)"
else
    warn "BT keyboard: returned $RESULT"
fi

# Test mouse
RESULT=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
    -d '{"x":1,"y":0}' "http://$PI_IP:8080/bt/mouse/move" 2>/dev/null || echo "000")
if [ "$RESULT" = "200" ]; then
    ok "BT mouse: working"
else
    warn "BT mouse: returned $RESULT"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo -e "\n${GREEN}=== All good ===${NC}"
echo ""
echo "  USB Ethernet:  $IFACE @ $MAC_IP -> $PI_IP"
echo "  REST API:      http://$PI_IP:8080  (unit: ${PI_UNIT:-unknown})"
echo "  BT HID:        connected"
echo ""
echo "  Test:  curl -X POST -H 'Content-Type: application/json' -d '{\"text\":\"hello\"}' http://$PI_IP:8080/bt/text"
echo ""
