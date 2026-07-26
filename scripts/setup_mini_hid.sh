#!/usr/bin/env bash
# setup_mini_hid.sh — turn a regular Linux host (a "mini") into the
# Bluetooth HID mouse/keyboard ("devmouse"), replacing the Raspberry Pi.
#
# This is the Ubuntu/desktop-Linux sibling of setup_bt_hid.sh: same BlueZ
# HID-peripheral configuration, minus everything Pi-specific (no
# pi-bluetooth package, no USB-gadget/ECM setup). The host's own
# Bluetooth radio becomes the emulated mouse the target Mac pairs with.
#
# Prerequisite: build the gateway venv first (as the normal user):
#   bash scripts/setup_mini_gateway_venv.sh
#
# Usage (run on the mini, as root):
#   sudo bash scripts/setup_mini_hid.sh
#
# Environment overrides (all auto-derived if unset):
#   MINI_USER   — the login user that owns the venv        (default: the sudo caller)
#   REPO_DIR    — handsneyes checkout root                 (default: derived from this script's path)
#   GW_VENV     — venv holding afferent-gateway + dbus      (default: $REPO_DIR/.venv-gw)
#
# Note: afferent-gateway binds port 8080 unconditionally (its entry point
# is main() with no argv/env port hook), so the port is fixed here.
#
# Stage 1 (this script): configure BlueZ, then launch the pairing agent
# and the afferent gateway as background root processes with logs under
# /tmp. Once pairing + mouse are verified end-to-end, promote them to
# systemd services (see the companion --install path, TODO).
set -uo pipefail

# --- resolve identity/paths (generalised — nothing hardcoded) --------------
# The user who invoked sudo (so we find *their* venv, not root's).
MINI_USER="${MINI_USER:-${SUDO_USER:-$(id -un)}}"
# Repo root = parent of this script's dir, resolved through symlinks.
_SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$_SELF")/.." && pwd)}"
GW_VENV="${GW_VENV:-$REPO_DIR/.venv-gw}"
GW_PORT=8080   # fixed: afferent-gateway's main() has no port override

GW_BIN="$GW_VENV/bin/afferent-gateway"
AGENT="$REPO_DIR/scripts/bt-agent.py"
CONF=/etc/bluetooth/main.conf
OVERRIDE_DIR=/etc/systemd/system/bluetooth.service.d
OVERRIDE=$OVERRIDE_DIR/override.conf

if [ "$(id -u)" -ne 0 ]; then
    echo "FATAL: run as root (sudo bash scripts/setup_mini_hid.sh)" >&2
    exit 1
fi
for p in "$GW_BIN" "$AGENT"; do
    [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done

echo "  user=$MINI_USER  repo=$REPO_DIR  venv=$GW_VENV  port=$GW_PORT"

echo "[1/6] Installing prerequisites (python3-gi/dbus/evdev)..."
apt-get install -y -qq python3-gi python3-dbus python3-evdev >/dev/null 2>&1 \
  && echo "  ok" || echo "  WARN apt install had issues (may already be present)"

echo "[2/6] Backing up + writing $CONF (Name=devmouse, HID class, bredr)..."
[ -f "$CONF" ] && [ ! -f "$CONF.pre-mini-hid" ] && cp "$CONF" "$CONF.pre-mini-hid"
cat > "$CONF" <<'BTCONF'
[General]
Name = devmouse
Class = 0x0025C0
ControllerMode = bredr
DiscoverableTimeout = 0
PairableTimeout = 0
Discoverable = true
FastConnectable = true

[Policy]
AutoEnable=true
BTCONF
echo "  ok"

echo "[3/6] bluetoothd override (--compat, disable input+audio plugins)..."
BTDAEMON=/usr/libexec/bluetooth/bluetoothd
[ -f "$BTDAEMON" ] || BTDAEMON=/usr/lib/bluetooth/bluetoothd
[ -f "$BTDAEMON" ] || { echo "  FATAL: bluetoothd not found" >&2; exit 1; }
mkdir -p "$OVERRIDE_DIR"
cat > "$OVERRIDE" <<EOF
[Service]
ExecStart=
ExecStart=$BTDAEMON --compat --noplugin=input,a2dp,avrcp,hfp,hsp,gateway,media,audio
EOF
echo "  daemon: $BTDAEMON"

echo "[4/6] Restarting bluetooth + asserting pairable/discoverable..."
systemctl daemon-reload
systemctl restart bluetooth
sleep 2
bluetoothctl power on          >/dev/null 2>&1 || true
bluetoothctl pairable on       >/dev/null 2>&1 || true
bluetoothctl discoverable on   >/dev/null 2>&1 || true
# The persisted per-adapter Alias overrides main.conf's Name for what the
# adapter advertises, so set it explicitly — otherwise the Mac sees the
# host's default hostname instead of "devmouse".
bluetoothctl system-alias devmouse >/dev/null 2>&1 || true
# Strip the audio/telephony SDP records bluetoothd publishes at the
# protocol level (--noplugin only filters plugin-level ones). Without
# this the Mac sees a "headset" and routes its system sound to the mini.
STRIP="$REPO_DIR/scripts/bt-strip-audio-sdp.sh"
[ -x "$STRIP" ] || STRIP="bash $REPO_DIR/scripts/bt-strip-audio-sdp.sh"
$STRIP >/dev/null 2>&1 && echo "  audio SDP records stripped" \
  || echo "  WARN audio-SDP strip had issues (run scripts/bt-strip-audio-sdp.sh manually)"
echo "  adapter: $(bluetoothctl show 2>/dev/null | grep -E 'Alias|Powered|Discoverable|Pairable' | tr '\n' ' ')"

echo "[5/6] Starting pairing agent (auto-accept, NoInputNoOutput)..."
pkill -f "bt-agent.py" 2>/dev/null || true
setsid python3 "$AGENT" >/tmp/mini-bt-agent.log 2>&1 < /dev/null &
sleep 1
pgrep -f "bt-agent.py" >/dev/null && echo "  agent up (log: /tmp/mini-bt-agent.log)" \
  || echo "  WARN agent not running — see /tmp/mini-bt-agent.log"

echo "[6/6] Starting afferent gateway as root (binds L2CAP PSM 17/19)..."
pkill -f "afferent-gateway" 2>/dev/null || true
sleep 1
setsid "$GW_BIN" >/tmp/mini-gateway.log 2>&1 < /dev/null &
sleep 3
if ss -ltnp 2>/dev/null | grep -q ":$GW_PORT"; then
  echo "  gateway up on :$GW_PORT (log: /tmp/mini-gateway.log)"
else
  echo "  WARN gateway not listening — see /tmp/mini-gateway.log"
fi

echo
echo "=== DONE (stage 1). Now, on the Mac: pair a NEW device 'devmouse'. ==="
echo "    Gateway log:  tail -f /tmp/mini-gateway.log"
echo "    Agent log:    tail -f /tmp/mini-bt-agent.log"
