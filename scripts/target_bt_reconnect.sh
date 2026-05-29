#!/usr/bin/env bash
# target_bt_reconnect.sh — keep the target's BT HID connection alive.
#
# Runs on the TARGET machine (the Ubuntu / Kali / macOS box the
# webcam watches). Polls every INTERVAL seconds; when bluetoothctl
# reports the Pi's BT HID device as paired-but-not-connected, runs
# `bluetoothctl connect`. Already-connected → no-op. Never-paired →
# logs a clear warning and keeps watching.
#
# Designed to run forever in a tmux pane or as a systemd --user
# unit. No daemonisation — keep it in the foreground so its output
# is inspectable.
#
# Modes:
#
#   ./target_bt_reconnect.sh             # loop forever
#   ./target_bt_reconnect.sh --once      # one check + exit (testing)
#   ./target_bt_reconnect.sh --probe     # print diagnostics + exit
#   ./target_bt_reconnect.sh --pair      # one-shot scan + pair + connect
#
# Auto-pairing: if no paired device matches PI_BT_NAMES, the main
# loop runs the same scan+pair+trust+connect sequence as `--pair`.
# Re-pairing relies on the Pi's auto-accept pairing agent
# (scripts/bt-agent.py on the Pi) — no user interaction needed on
# the target. SCAN_TIMEOUT bounds the discovery phase.
#
# Configuration via env vars (see also `--probe` to see what the
# script sees on this host):
#
#   PI_BT_NAMES   Comma-separated list of names to look up.
#                 Default: "keyboarder,devmouse"
#   PI_BT_MAC     Explicit MAC, e.g. AA:BB:CC:DD:EE:FF — skips the
#                 by-name lookup. Use this when bluetoothctl can't
#                 find the device by name.
#   INTERVAL      Seconds between checks. Default: 300 (5 min).
#   SCAN_TIMEOUT  Seconds to scan when re-pairing. Default: 25.
#   LOG_FILE      Optional path; output is also appended there.
#   DEBUG         Set to 1 to print every bluetoothctl response.

set -u

# ── config ─────────────────────────────────────────────────────────
PI_BT_NAMES="${PI_BT_NAMES:-keyboarder,devmouse}"
INTERVAL="${INTERVAL:-300}"
SCAN_TIMEOUT="${SCAN_TIMEOUT:-25}"
LOG_FILE="${LOG_FILE:-}"
DEBUG="${DEBUG:-0}"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() {
    local msg="[$(ts)] $*"
    echo "$msg"
    [ -n "$LOG_FILE" ] && echo "$msg" >> "$LOG_FILE"
}
debug() { [ "$DEBUG" = "1" ] && log "  [debug] $*" || true; }

# ── bluetoothctl helpers ───────────────────────────────────────────
btc() {
    # Run bluetoothctl with a hard timeout. Returns the trimmed
    # output. Bluetoothctl sometimes hangs in non-interactive mode
    # on lossy adapters; the timeout keeps us moving.
    timeout 12 bluetoothctl "$@" 2>&1
}

# Discover paired devices. Tries newer ``devices Paired`` syntax
# first (bluez >= 5.65), falls back to ``paired-devices`` (legacy
# until bluez 5.65 deprecation), then to ``devices`` filtered by
# our list of accepted names. Returns lines of the form
# "<MAC> <name>".
list_paired() {
    local out
    # 1) Modern: `devices Paired`
    out=$(btc devices Paired 2>/dev/null)
    if echo "$out" | grep -qE '^Device [0-9A-F:]{17}'; then
        echo "$out" | awk '/^Device / {
            mac=$2; $1=""; $2=""; sub(/^  */,"")
            print mac" "$0
        }'
        return
    fi
    # 2) Legacy: `paired-devices`
    out=$(btc paired-devices 2>/dev/null)
    if echo "$out" | grep -qE '^Device [0-9A-F:]{17}'; then
        echo "$out" | awk '/^Device / {
            mac=$2; $1=""; $2=""; sub(/^  */,"")
            print mac" "$0
        }'
        return
    fi
    # 3) Last resort: list all devices (not just paired); we'll
    #    cross-check Paired=yes via `info` below.
    out=$(btc devices 2>/dev/null)
    if echo "$out" | grep -qE '^Device [0-9A-F:]{17}'; then
        echo "$out" | awk '/^Device / {
            mac=$2; $1=""; $2=""; sub(/^  */,"")
            print mac" "$0
        }'
        return
    fi
}

# Resolve the Pi's MAC. Tries each comma-separated name from
# PI_BT_NAMES against the paired list (case-insensitive substring).
# Returns the first MAC that ALSO reports ``Paired: yes`` in
# `bluetoothctl info`.
find_mac() {
    local paired
    paired=$(list_paired)
    debug "list_paired: $(echo "$paired" | wc -l) entries"
    [ -z "$paired" ] && return 1

    local IFS=','
    for raw_name in $PI_BT_NAMES; do
        local name
        name=$(echo "$raw_name" | sed 's/^ *//;s/ *$//')
        [ -z "$name" ] && continue
        debug "looking for $name"
        local line
        line=$(echo "$paired" \
            | grep -i -F "$name" \
            | head -n 1)
        if [ -n "$line" ]; then
            local mac
            mac=$(echo "$line" | awk '{print $1}')
            # Confirm it's actually paired (some bluez versions
            # list discovered-but-unpaired devices too).
            if btc info "$mac" 2>/dev/null | grep -q "Paired: yes"; then
                echo "$mac"
                return 0
            fi
            debug "$name -> $mac is not Paired:yes; skipping"
        fi
    done
    return 1
}

is_connected() {
    local mac="$1"
    btc info "$mac" 2>/dev/null | grep -q "Connected: yes"
}

# ── pair flow ──────────────────────────────────────────────────────
# Used when find_mac returned nothing — try to discover the Pi by
# name via an active scan, then pair + trust + connect. Relies on
# the Pi running an auto-accept agent (scripts/bt-agent.py).

# Scan-once: turn scanning on, wait SCAN_TIMEOUT, turn off, then
# read the discovered-device list. We use a background scanning
# session so we can interrogate `devices` while the scan is active.
scan_for_pi() {
    log "scanning for Pi (up to ${SCAN_TIMEOUT}s)..."
    btc power on  >/dev/null 2>&1
    btc agent on  >/dev/null 2>&1
    btc default-agent >/dev/null 2>&1
    # Start a background scan. We DON'T want to use `timeout` with
    # `bluetoothctl scan on` because that exits non-zero on TERM and
    # bluetoothctl prints to a pty; instead, send the command in the
    # background to a subshell.
    btc --timeout "$SCAN_TIMEOUT" scan on >/dev/null 2>&1 &
    local scan_pid=$!

    local mac=""
    local elapsed=0
    while [ "$elapsed" -lt "$SCAN_TIMEOUT" ] && [ -z "$mac" ]; do
        sleep 3
        elapsed=$((elapsed + 3))
        mac=$(find_mac 2>/dev/null || true)
        if [ -z "$mac" ]; then
            # Also scan the open `devices` list (includes unpaired
            # discoveries) for any of our names.
            local all
            all=$(btc devices 2>/dev/null)
            local IFS=','
            for raw_name in $PI_BT_NAMES; do
                local name
                name=$(echo "$raw_name" | sed 's/^ *//;s/ *$//')
                [ -z "$name" ] && continue
                local hit
                hit=$(echo "$all" | grep -i -F "$name" | head -n 1)
                if [ -n "$hit" ]; then
                    mac=$(echo "$hit" | awk '{print $2}')
                    log "  discovered (unpaired): $mac ($name)"
                    break
                fi
            done
        fi
    done
    # Stop the background scan.
    btc scan off >/dev/null 2>&1
    wait "$scan_pid" 2>/dev/null || true
    if [ -z "$mac" ]; then
        log "  ✗ scan timed out; Pi not discoverable"
        return 1
    fi
    echo "$mac"
}

pair_and_connect() {
    local mac="$1"
    log "pairing with $mac..."
    btc power on  >/dev/null 2>&1
    btc agent on  >/dev/null 2>&1
    btc default-agent >/dev/null 2>&1
    local pair_out
    pair_out=$(btc pair "$mac" 2>&1 || true)
    debug "pair output: $pair_out"
    if echo "$pair_out" | grep -qE "(Pairing successful|already-paired|AlreadyExists)"; then
        log "  ✓ paired"
    else
        log "  ✗ pair failed: $(echo "$pair_out" | tail -1)"
        return 1
    fi
    btc trust "$mac"   >/dev/null 2>&1
    attempt_connect "$mac"
    return 0
}

attempt_connect() {
    local mac="$1"
    log "Pi BT HID ($mac) disconnected — attempting reconnect..."
    # Belt-and-braces: ensure adapter is powered + trust the device
    # before connecting. A bare ``connect`` silently fails when
    # the adapter is soft-blocked / not powered / device untrusted.
    btc power on    >/dev/null 2>&1
    btc agent on    >/dev/null 2>&1
    btc trust "$mac" >/dev/null 2>&1
    local result
    result=$(btc connect "$mac" 2>&1 || true)
    debug "connect output: $result"
    if echo "$result" | grep -q "Connection successful"; then
        log "  ✓ connected"
    elif echo "$result" | grep -q "already-connected"; then
        log "  ✓ already connected (no-op)"
    elif echo "$result" | grep -q "page-timeout"; then
        log "  ✗ page-timeout (Pi adapter unreachable / off)"
    elif echo "$result" | grep -q "create-socket"; then
        log "  ✗ create-socket (Pi service not serving L2CAP yet)"
    else
        # Last line is usually the most useful error.
        log "  ✗ connect failed: $(echo "$result" | tail -1)"
    fi
}

# ── probe (diagnostic, no loop) ────────────────────────────────────
probe() {
    log "── probe: what does bluetoothctl see on this host? ──"
    log "bluetoothctl --version:"
    btc --version | sed 's/^/    /'
    log "bluetoothctl show:"
    btc show 2>&1 | head -20 | sed 's/^/    /'
    log "list_paired output:"
    list_paired | sed 's/^/    /' || log "    (no paired devices found)"
    log "looking for names: $PI_BT_NAMES"
    local mac
    if mac=$(find_mac); then
        log "  ✓ matched MAC: $mac"
        log "  info $mac:"
        btc info "$mac" 2>&1 | sed 's/^/    /' | head -25
    else
        log "  ✗ no match. Pair the Pi via the GNOME / KDE / macOS"
        log "    BT settings panel first; the script will not pair"
        log "    a brand-new device for you (initial pairing needs"
        log "    a user-side confirmation)."
    fi
}

# ── main loop ──────────────────────────────────────────────────────
main_loop() {
    local one_shot="$1"
    local mac="${PI_BT_MAC:-}"

    if [ -z "$mac" ]; then
        mac=$(find_mac || true)
    fi
    if [ -z "$mac" ]; then
        log "no paired device matched names [$PI_BT_NAMES]."
        log "will attempt scan+pair on next iteration."
    else
        log "monitoring $mac, every ${INTERVAL}s "
        log "(names tried: $PI_BT_NAMES)"
    fi

    while true; do
        # Re-discover the MAC each loop so a freshly-paired device
        # gets picked up without restarting the script.
        if [ -z "$mac" ]; then
            mac=$(find_mac || true)
            [ -n "$mac" ] && log "discovered (paired) MAC: $mac"
        fi

        if [ -n "$mac" ]; then
            if is_connected "$mac"; then
                debug "$mac already connected — nothing to do"
            else
                attempt_connect "$mac"
            fi
        else
            # No paired MAC. Try a scan+pair cycle. If the scan
            # finds the Pi but pairing fails (e.g. Pi agent not
            # running), we fall back to retry on the next loop.
            local scanned
            if scanned=$(scan_for_pi); then
                pair_and_connect "$scanned" && mac="$scanned"
            fi
        fi

        [ "$one_shot" = "1" ] && return 0
        sleep "$INTERVAL"
    done
}

# Explicit one-shot: scan + pair + connect, regardless of current
# state. Useful when the Pi was forgotten on the target and the
# user wants the script to re-establish the pairing immediately
# (instead of waiting for the next INTERVAL).
pair_once() {
    local mac="${PI_BT_MAC:-}"
    if [ -z "$mac" ]; then
        mac=$(find_mac || true)
    fi
    if [ -z "$mac" ]; then
        if ! mac=$(scan_for_pi); then
            log "  ✗ could not discover Pi by name; aborting"
            return 1
        fi
    fi
    pair_and_connect "$mac"
}

# ── entry ──────────────────────────────────────────────────────────
# macOS uses an entirely different BT stack (no bluetoothctl). Detect
# early so the user gets a useful pointer instead of "command not
# found" 200 lines into a Linux-only script.
if [ "$(uname)" = "Darwin" ]; then
    echo "This script is the Linux/BlueZ variant. macOS uses a"
    echo "different Bluetooth stack — run the macOS variant instead:"
    echo "  ./scripts/target_bt_reconnect_macos.sh ${*:-}"
    exit 1
fi
command -v bluetoothctl >/dev/null 2>&1 || {
    echo "ERROR: bluetoothctl not found. Install bluez:"
    echo "  Debian/Ubuntu/Kali: sudo apt install -y bluez"
    echo "  Fedora:             sudo dnf install -y bluez"
    echo "  Arch:               sudo pacman -S bluez bluez-utils"
    exit 1
}

case "${1:-}" in
    --probe)  probe ;;
    --pair)   pair_once ;;
    --once)   main_loop 1 ;;
    "")       main_loop 0 ;;
    *)
        echo "usage: $0 [--probe | --pair | --once]"
        exit 2
        ;;
esac
