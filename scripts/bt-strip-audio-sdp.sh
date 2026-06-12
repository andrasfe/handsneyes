#!/usr/bin/env bash
# bt-strip-audio-sdp.sh — Remove default audio/telephony SDP records
# AND make sure the adapter is in a connectable state.
#
# BlueZ 5.x publishes Hands-Free, Audio Gateway, SIM Access, and
# similar SDP records automatically as part of its protocol stack,
# regardless of --noplugin flags (those are at protocol level, not
# plugin level).  When the Pi is paired as a HID device the host (a
# Mac, in our case) sees those audio-profile records and silently
# adds the Pi as a Bluetooth audio output — then routes the host's
# system sound there.  The Pi has no DAC and no speaker driver, so
# the audio goes nowhere and the operator loses Mac speaker audio
# every time the Pi connects.
#
# This script ALSO restores the adapter's runtime settings that
# bluetoothd doesn't reliably re-apply from main.conf on a hot
# restart: Pairable=yes (the load-bearing one — without it macOS
# pair requests get silently rejected), Discoverable=yes, and the
# user-visible Alias=devmouse.  hciconfig name is forced too because
# bluetoothd falls back to the Pi's hostname for the GAP Name field
# even when main.conf sets one.
#
# Runs after bluetoothd starts (driven by a systemd Wants= /
# PartOf= on bluetooth.service).  bluetoothd keeps the HID record
# we registered through RegisterProfile + the standard GAP / GATT /
# Device Information records — exactly what a HID-only peer should
# advertise.

set -u

UNWANTED_NAMES=(
    "Hands-Free Voice gateway"
    "Hands-Free unit"
    "Headset"
    "Headset Audio Gateway"
    "Audio Source"
    "Audio Sink"
    "AV Remote Control"
    "AV Remote Control Target"
    "SIM Access Server"
    "Message Notification Server"
    "Message Access Server"
    "Phonebook Access"
    "Phonebook Access Server"
)

DELETED=0
SDPDATA=$(sdptool browse local 2>&1)

for NAME in "${UNWANTED_NAMES[@]}"; do
    # Find handles for all records whose Service Name matches NAME.
    # sdptool output puts "Service RecHandle:" immediately after
    # "Service Name:".  Loop over each match in case the same name
    # appears in multiple records.
    while read -r HANDLE; do
        [ -z "$HANDLE" ] && continue
        if sdptool del "$HANDLE" 2>&1 | grep -q deleted; then
            echo "  stripped '$NAME' ($HANDLE)"
            DELETED=$((DELETED + 1))
        fi
    done < <(
        echo "$SDPDATA" | awk -v name="Service Name: $NAME" '
            $0 == name { next_line = 1; next }
            next_line && /^Service RecHandle:/ {
                print $3
                next_line = 0
            }
        '
    )
done

echo "bt-strip-audio-sdp: removed $DELETED audio/telephony record(s)."

# ------------------------------------------------------------------
# Force adapter into a connectable, pairable state.
# main.conf's Pairable= isn't reliably honoured on hot restart, and
# the user-visible Alias gets reset to the hostname.  Setting these
# from the script makes them deterministic regardless of how bluetoothd
# came up.
# ------------------------------------------------------------------
hciconfig hci0 name "devmouse" 2>/dev/null || true

bluetoothctl <<'BCTL' >/dev/null 2>&1
power on
discoverable on
pairable on
system-alias devmouse
BCTL

# Quick verify line for the journal
PAIR_STATE=$(bluetoothctl show 2>/dev/null | awk -F': ' '/Pairable/ {print $2}' | head -1)
ALIAS_NOW=$(bluetoothctl show 2>/dev/null | awk -F': ' '/Alias/ {print $2}' | head -1)
echo "bt-strip-audio-sdp: pairable=$PAIR_STATE alias=$ALIAS_NOW"
