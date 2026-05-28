#!/usr/bin/env bash
# target_bt_reconnect_macos.sh — keep the target's BT HID connection
# alive on macOS.
#
# macOS variant of target_bt_reconnect.sh. Same semantics: polls every
# INTERVAL seconds; when the Pi's BT HID is paired-but-not-connected,
# reconnects. Already-connected → no-op. Never-paired → scan + pair
# (relies on the Pi's auto-accept agent for the handshake).
#
# Uses `blueutil` instead of `bluetoothctl` (BlueZ tool, Linux-only).
# Install once:  brew install blueutil
#
# Modes:
#
#   ./target_bt_reconnect_macos.sh              # loop forever
#   ./target_bt_reconnect_macos.sh --once       # one check + exit
#   ./target_bt_reconnect_macos.sh --probe      # diagnostics + exit
#   ./target_bt_reconnect_macos.sh --pair       # one-shot scan + pair + connect
#
# Configuration via env vars (mirrors the Linux variant):
#
#   PI_BT_MAC      Explicit MAC of the Pi (preferred — macOS BT names
#                  are less reliable than Linux). Format like:
#                  B8:27:EB:E7:2B:70  or  b827ebe72b70 (blueutil
#                  accepts both; we normalise to the dashed form).
#   PI_BT_NAMES    Comma-separated name fallback when PI_BT_MAC isn't
#                  set. Default: "keyboarder,handsneyes HID"
#   INTERVAL       Seconds between checks. Default: 300 (5 min).
#   SCAN_TIMEOUT   Seconds to scan when discovering for pair. Default: 25.
#   LOG_FILE       Optional path; output is also appended there.

set -u

PI_BT_NAMES="${PI_BT_NAMES:-keyboarder,handsneyes HID}"
PI_BT_MAC="${PI_BT_MAC:-}"
INTERVAL="${INTERVAL:-300}"
SCAN_TIMEOUT="${SCAN_TIMEOUT:-25}"
LOG_FILE="${LOG_FILE:-}"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() {
    local line="$(ts) $*"
    echo "$line"
    [ -n "$LOG_FILE" ] && echo "$line" >> "$LOG_FILE"
}

# Normalise MAC: accept either "B8:27:EB:E7:2B:70" or "b827ebe72b70".
# blueutil's --info accepts both, but its output (and --paired/--connect)
# emits the dashed lowercase form, which we use internally.
normalise_mac() {
    local m="${1:-}"
    if [ -z "$m" ]; then return 1; fi
    # Strip everything that isn't a hex char, then re-insert dashes.
    local clean
    clean=$(echo "$m" | tr 'A-F' 'a-f' | tr -d -c '0-9a-f')
    if [ ${#clean} -ne 12 ]; then
        return 1
    fi
    echo "${clean:0:2}:${clean:2:2}:${clean:4:2}:${clean:6:2}:${clean:8:2}:${clean:10:2}"
}

# Look up the Pi's MAC by name from the list of paired devices. Returns
# the dashed-lowercase MAC on stdout, or empty string on miss.
mac_from_name() {
    local want_name="$1"
    blueutil --paired --format json 2>/dev/null \
        | python3 -c "
import json, sys
want = sys.argv[1].lower()
for d in json.load(sys.stdin):
    if (d.get('name') or '').lower() == want:
        print(d.get('address',''))
        break
" "$want_name"
}

# Resolve the target's MAC: explicit PI_BT_MAC wins; otherwise try each
# name in PI_BT_NAMES.
resolve_mac() {
    if [ -n "$PI_BT_MAC" ]; then
        local m
        m=$(normalise_mac "$PI_BT_MAC")
        if [ -n "$m" ]; then echo "$m"; return 0; fi
        log "WARN: PI_BT_MAC=$PI_BT_MAC didn't parse to a 12-hex MAC."
    fi
    local IFS=,
    for n in $PI_BT_NAMES; do
        n=$(echo "$n" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$n" ] && continue
        local m
        m=$(mac_from_name "$n")
        if [ -n "$m" ]; then
            echo "$m"
            return 0
        fi
    done
    return 1
}

is_connected() {
    local mac="$1"
    # `blueutil --is-connected <mac>` prints "1" or "0".
    [ "$(blueutil --is-connected "$mac" 2>/dev/null)" = "1" ]
}

is_paired() {
    local mac="$1"
    blueutil --paired --format json 2>/dev/null \
        | python3 -c "
import json, sys
mac = sys.argv[1].lower()
hits = [d for d in json.load(sys.stdin) if (d.get('address') or '').lower() == mac]
sys.exit(0 if hits else 1)
" "$mac"
}

# One-shot connect attempt. Returns 0 on success, 1 on failure.
try_connect() {
    local mac="$1"
    log "connecting to $mac …"
    if blueutil --connect "$mac" >/dev/null 2>&1; then
        sleep 1
        if is_connected "$mac"; then
            log "OK — connected to $mac"
            return 0
        fi
        log "  blueutil returned 0 but is-connected says 0; will retry next tick"
        return 1
    fi
    log "  blueutil --connect failed (rc=$?)"
    return 1
}

scan_for_target() {
    log "scanning for $SCAN_TIMEOUT s …"
    # blueutil --inquiry runs a fixed-duration scan and prints results.
    local results
    results=$(blueutil --inquiry "$SCAN_TIMEOUT" --format json 2>/dev/null || echo "[]")
    if [ -z "$PI_BT_MAC" ]; then
        # Pick the first scan result whose name matches PI_BT_NAMES.
        local IFS=,
        for n in $PI_BT_NAMES; do
            n=$(echo "$n" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            [ -z "$n" ] && continue
            local m
            m=$(echo "$results" | python3 -c "
import json, sys
want = sys.argv[1].lower()
data = json.load(sys.stdin)
for d in data:
    if (d.get('name') or '').lower() == want:
        print(d.get('address',''))
        break
" "$n")
            if [ -n "$m" ]; then echo "$m"; return 0; fi
        done
    else
        # Verify the explicit MAC was visible.
        local m
        m=$(normalise_mac "$PI_BT_MAC")
        local seen
        seen=$(echo "$results" | python3 -c "
import json, sys
mac = sys.argv[1].lower()
hits = [d for d in json.load(sys.stdin) if (d.get('address') or '').lower() == mac]
print('y' if hits else 'n')
" "$m")
        if [ "$seen" = "y" ]; then echo "$m"; return 0; fi
    fi
    return 1
}

pair_and_connect() {
    local mac="$1"
    if ! is_paired "$mac"; then
        log "pairing $mac …"
        if ! blueutil --pair "$mac" 2>&1 | sed 's/^/  /'; then
            log "  pair failed"
            return 1
        fi
    fi
    try_connect "$mac"
}

pair_once() {
    local mac
    if [ -n "$PI_BT_MAC" ]; then
        mac=$(normalise_mac "$PI_BT_MAC")
    else
        mac=$(scan_for_target) || {
            log "scan did not surface any of $PI_BT_NAMES — abort"
            exit 2
        }
    fi
    pair_and_connect "$mac" || exit 3
}

probe() {
    log "── probe: what does blueutil see on this host? ──"
    log "blueutil --version: $(blueutil --version 2>&1)"
    log ""
    log "blueutil --power: $(blueutil --power 2>&1)"
    log ""
    log "Paired devices:"
    blueutil --paired --format json 2>/dev/null \
        | python3 -m json.tool 2>/dev/null \
        | sed 's/^/  /' || log "  (none / parse failed)"
    log ""
    local mac
    if mac=$(resolve_mac); then
        log "Resolved Pi MAC: $mac"
        log "Connected: $(blueutil --is-connected "$mac" 2>/dev/null)"
    else
        log "Pi MAC not resolvable from PI_BT_MAC=$PI_BT_MAC PI_BT_NAMES=$PI_BT_NAMES"
    fi
}

main_loop() {
    local once_only="$1"
    while true; do
        local mac
        if mac=$(resolve_mac); then
            if is_connected "$mac"; then
                log "OK — $mac already connected"
            else
                try_connect "$mac" || {
                    if ! is_paired "$mac"; then
                        log "  not paired; attempting one-shot pair-and-connect"
                        pair_and_connect "$mac"
                    fi
                }
            fi
        else
            log "Pi not in paired devices; running --pair flow"
            pair_once || log "  pair flow failed; will retry next tick"
        fi
        [ "$once_only" = "1" ] && break
        sleep "$INTERVAL"
    done
}

# ── entry ──────────────────────────────────────────────────────────
if [ "$(uname)" != "Darwin" ]; then
    echo "This script is the macOS variant. On Linux use:"
    echo "  ./scripts/target_bt_reconnect.sh ${*:-}"
    exit 1
fi
command -v blueutil >/dev/null 2>&1 || {
    echo "ERROR: blueutil not found. Install via Homebrew:"
    echo "  brew install blueutil"
    echo ""
    echo "blueutil is the CLI bridge to macOS' IOBluetooth framework."
    echo "It needs Bluetooth + Accessibility privacy permissions the"
    echo "first time it runs — macOS will prompt."
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
