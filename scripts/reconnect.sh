#!/usr/bin/env bash
# reconnect.sh — Restore USB ECM + BT HID connectivity after cable change or reboot.
# Run from the dev Mac. No arguments needed. Guides you through every step.

set -euo pipefail

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

check_bt() {
    local health bt
    health=$(curl -s --connect-timeout 2 "http://$PI_IP:8080/health" 2>/dev/null || echo "{}")
    bt=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('bt_hid_connected',False))" 2>/dev/null || echo "False")
    [ "$bt" = "True" ]
}

# ===========================================================================
# Step 1: Find and configure USB Ethernet interface
# ===========================================================================
echo -e "\n${BLUE}=== Step 1: USB Ethernet ===${NC}"

ATTEMPT=0
while true; do
    IFACE=$(ifconfig -l | tr ' ' '\n' | while read if; do
        ifconfig "$if" 2>/dev/null | grep -q "48:6f:73:74" && echo "$if" && break
    done)

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
CURRENT_IP=$(ifconfig "$IFACE" 2>/dev/null | grep "inet " | awk '{print $2}')
if [ "$CURRENT_IP" = "$MAC_IP" ]; then
    ok "IP already set: $MAC_IP"
else
    info "Setting $IFACE to $MAC_IP ..."
    sudo ifconfig "$IFACE" "$MAC_IP" netmask 255.255.255.0 up
    sleep 1
    ok "IP configured: $MAC_IP on $IFACE"
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
# Step 3: Check services on Pi
# ===========================================================================
echo -e "\n${BLUE}=== Step 3: Pi services ===${NC}"

# Check / start handsneyes-pi
if $SSH "$PI_USER@$PI_IP" "pgrep -f handsneyes-pi" > /dev/null 2>&1; then
    ok "handsneyes-pi running"
else
    info "Starting handsneyes-pi ..."
    $SSH "$PI_USER@$PI_IP" "echo 'andras' | sudo -S systemctl start handsneyes-pi" 2>/dev/null || true
    sleep 5
    if $SSH "$PI_USER@$PI_IP" "pgrep -f handsneyes-pi" > /dev/null 2>&1; then
        ok "handsneyes-pi started"
    else
        fail "Could not start handsneyes-pi"
        echo "  Check logs: ssh $PI_USER@$PI_IP 'sudo journalctl -u handsneyes-pi --since \"5 min ago\"'"
        exit 1
    fi
fi

# Wait for REST API
info "Waiting for REST API ..."
TRIES=0
while ! curl -s --connect-timeout 2 "http://$PI_IP:8080/health" > /dev/null 2>&1; do
    TRIES=$((TRIES + 1))
    if [ "$TRIES" -ge 15 ]; then
        fail "REST API not responding after 30s"
        echo "  Try restarting: ssh $PI_USER@$PI_IP 'sudo systemctl restart handsneyes-pi'"
        exit 1
    fi
    sleep 2
done
ok "REST API responding"

# Check / start pairing agent
if $SSH "$PI_USER@$PI_IP" "pgrep -f bt-agent.py" > /dev/null 2>&1; then
    ok "Pairing agent running"
else
    info "Starting pairing agent ..."
    $SSH "$PI_USER@$PI_IP" "echo 'andras' | sudo -S bash -c 'PYTHONUNBUFFERED=1 setsid python3 /home/andras/handsneyes/scripts/bt-agent.py > /tmp/bt-agent.log 2>&1 < /dev/null &'" 2>/dev/null || true
    sleep 3
    if $SSH "$PI_USER@$PI_IP" "pgrep -f bt-agent.py" > /dev/null 2>&1; then
        ok "Pairing agent started"
    else
        warn "Pairing agent failed to start — pairing may hang"
        echo "  Try manually: ssh $PI_USER@$PI_IP 'sudo python3 ~/handsneyes/scripts/bt-agent.py'"
    fi
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
            echo "  On the TARGET Mac:"
            echo "    1. Open System Settings → Bluetooth"
            echo "    2. Look for 'handsneyes HID' or 'keyboarder' in Nearby Devices"
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
            warn "Still not connected — restarting BT on Pi"
            echo ""
            $SSH "$PI_USER@$PI_IP" "echo 'andras' | sudo -S bash -c 'systemctl restart bluetooth; sleep 2; hciconfig hci0 up; hciconfig hci0 class 0x0025C0; hciconfig hci0 piscan; systemctl restart handsneyes-pi'" 2>/dev/null || true
            sleep 5
            # Restart agent (bluetooth restart kills it)
            $SSH "$PI_USER@$PI_IP" "echo 'andras' | sudo -S bash -c 'pkill -f bt-agent.py 2>/dev/null; sleep 1; PYTHONUNBUFFERED=1 setsid python3 /home/andras/handsneyes/scripts/bt-agent.py > /tmp/bt-agent.log 2>&1 < /dev/null &'" 2>/dev/null || true
            sleep 3
            echo "  Bluetooth restarted on Pi."
            echo "  On the TARGET Mac:"
            echo "    1. Forget 'handsneyes HID' / 'keyboarder' if listed"
            echo "    2. Toggle Bluetooth off and on"
            echo "    3. Wait for device to appear, then Connect"
        else
            warn "Still not connected — clean slate"
            echo ""
            echo "  Removing all pairings on Pi and doing full reset..."
            $SSH "$PI_USER@$PI_IP" "echo 'andras' | sudo -S bash -c '
                for dev in \$(echo \"devices\" | bluetoothctl 2>/dev/null | grep Device | awk \"{print \\\$2}\"); do
                    echo \"remove \$dev\" | bluetoothctl 2>/dev/null
                done
                systemctl restart bluetooth
                sleep 2
                hciconfig hci0 up
                hciconfig hci0 class 0x0025C0
                hciconfig hci0 piscan
                systemctl restart handsneyes-pi
            '" 2>/dev/null || true
            sleep 5
            $SSH "$PI_USER@$PI_IP" "echo 'andras' | sudo -S bash -c 'pkill -f bt-agent.py 2>/dev/null; sleep 1; PYTHONUNBUFFERED=1 setsid python3 /home/andras/handsneyes/scripts/bt-agent.py > /tmp/bt-agent.log 2>&1 < /dev/null &'" 2>/dev/null || true
            sleep 3
            echo "  Full reset done."
            echo "  On the TARGET Mac:"
            echo "    1. Forget 'handsneyes HID' / 'keyboarder' if listed"
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

HEALTH=$(curl -s "http://$PI_IP:8080/health" 2>/dev/null)

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
echo "  REST API:      http://$PI_IP:8080"
echo "  BT HID:        connected"
echo ""
echo "  Test:  curl -X POST -H 'Content-Type: application/json' -d '{\"text\":\"hello\"}' http://$PI_IP:8080/bt/text"
echo ""
