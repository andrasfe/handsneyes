#!/usr/bin/env bash
# install_local_hid_services.sh — make the local BT-HID host survive reboots.
#
# setup_local_hid.sh brings the stack up *now* by launching the pairing agent
# and the gateway as ad-hoc background root processes. Those die on reboot (and
# their /tmp logs are cleared), so the mouse silently stops working until
# someone re-runs the script by hand. This installs both as systemd services
# with Restart=always, so a reboot — or a crash — restores them automatically.
#
#   sudo bash scripts/install_local_hid_services.sh            # install + start
#   sudo bash scripts/install_local_hid_services.sh --uninstall
#
# Status / logs afterwards:
#   systemctl status handsneyes-gateway handsneyes-btagent
#   journalctl -u handsneyes-gateway -f
set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$SELF")/.." && pwd)}"
GW_VENV="${GW_VENV:-$REPO_DIR/.venv-gw}"
GW_BIN="$GW_VENV/bin/afferent-gateway"
AGENT="$REPO_DIR/scripts/bt-agent.py"
ALIAS="${BT_ALIAS:-devmouse}"

GW_UNIT=/etc/systemd/system/handsneyes-gateway.service
AGENT_UNIT=/etc/systemd/system/handsneyes-btagent.service
ASSERT=/usr/local/bin/handsneyes-bt-assert.sh

[ "$(id -u)" -eq 0 ] || { echo "FATAL: run with sudo" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
    systemctl disable --now handsneyes-gateway handsneyes-btagent 2>/dev/null
    rm -f "$GW_UNIT" "$AGENT_UNIT" "$ASSERT"
    systemctl daemon-reload
    echo "removed handsneyes-gateway + handsneyes-btagent"
    exit 0
fi

for p in "$GW_BIN" "$AGENT"; do
    [ -e "$p" ] || { echo "FATAL: missing $p (run setup_local_gateway_venv.sh first)" >&2; exit 1; }
done

# BlueZ 5.8x ignores `Pairable` in main.conf, and the runtime alias must be
# re-asserted after bluetoothd starts — otherwise the adapter comes back
# advertising the host's hostname.
#
# Discoverable/pairable are only turned on while NOTHING is bonded yet, i.e.
# during first pairing. Leaving them on permanently makes every other machine
# in range see "devmouse" and pop up pairing prompts — and a bonded target
# reconnects fine without them.
cat > "$ASSERT" <<EOF
#!/usr/bin/env bash
sleep 2
bluetoothctl power on            >/dev/null 2>&1 || true
bluetoothctl system-alias $ALIAS >/dev/null 2>&1 || true
# Force the HID device class (peripheral / combo keyboard+pointing).
# BlueZ does not reliably apply main.conf's Class= — it derives the class
# from registered profiles, so with the input plugin disabled (and no audio
# endpoints registered) the adapter falls back to 0x000104 "Computer".
# A target scanning for a mouse then does not show it as one.
hciconfig hci0 class 0x0025c0 >/dev/null 2>&1 || true
if [ -z "\$(bluetoothctl devices Paired 2>/dev/null)" ]; then
    # No target bonded yet — advertise so it can be paired.
    bluetoothctl pairable on     >/dev/null 2>&1 || true
    bluetoothctl discoverable on >/dev/null 2>&1 || true
else
    # Already bonded: stay quiet so other machines stop prompting.
    bluetoothctl pairable off     >/dev/null 2>&1 || true
    bluetoothctl discoverable off >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod +x "$ASSERT"

cat > "$AGENT_UNIT" <<EOF
[Unit]
Description=handsneyes Bluetooth pairing agent (auto-accept)
After=bluetooth.service
PartOf=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 $AGENT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > "$GW_UNIT" <<EOF
[Unit]
Description=handsneyes BT-HID gateway (afferent)
After=bluetooth.service handsneyes-btagent.service
Requires=bluetooth.service

[Service]
Type=simple
# Re-assert the adapter's runtime state before the gateway binds L2CAP.
ExecStartPre=$ASSERT
ExecStart=$GW_BIN
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now handsneyes-btagent handsneyes-gateway 2>&1 | tail -2
sleep 4

echo
for u in handsneyes-btagent handsneyes-gateway; do
    printf '%-24s %s\n' "$u" "$(systemctl is-active $u)"
done
if ss -ltn 2>/dev/null | grep -q ':8080'; then
    echo "gateway listening on :8080 — will now survive reboots"
else
    echo "WARN gateway not listening yet — journalctl -u handsneyes-gateway -n 30"
fi
