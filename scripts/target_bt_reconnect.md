# `target_bt_reconnect.sh`

Keeps the **target machine's** Bluetooth HID connection to the Pi
alive without manual intervention. Polls every few minutes; when the
Pi's HID device is paired-but-not-connected, runs
`bluetoothctl connect`. Already-connected? No-op. Never-paired?
Warns and keeps watching.

Symptom this fixes: every time the Pi service restarts
(`systemctl restart handsneyes-pi`) the L2CAP sockets close, the
target sees the BT HID device disappear, and the target's BT stack
doesn't always auto-reconnect. Without this script you have to
manually toggle Bluetooth or click "Connect" each time.

The script also handles the harder case: if pairing has been lost
entirely (you cleared the device, the target was reset, etc.), it
will **scan + pair + trust + connect** on its own — no manual
clicking. This depends on the Pi running its auto-accept pairing
agent (`scripts/bt-agent.py` on the Pi side, see project README);
the script registers a local `NoInputNoOutput` agent on the target
so neither side prompts the user.

The script runs on the **target** machine (the Ubuntu / Kali / Mac
that the webcam is watching). It does not run on the dev Mac, and
does not run on the Pi.

---

## Pick the variant for your target OS

| target OS | script | required CLI |
|---|---|---|
| Linux (Ubuntu / Debian / Kali / Fedora / Arch) | `target_bt_reconnect.sh`        | `bluetoothctl` (BlueZ) |
| macOS                                          | `target_bt_reconnect_macos.sh`  | `blueutil` (brew install blueutil) |

The two have the same flags (`--once`, `--pair`, `--probe`) and the
same env vars (`PI_BT_MAC`, `PI_BT_NAMES`, `INTERVAL`, `SCAN_TIMEOUT`,
`LOG_FILE`). Running either one on the wrong OS prints a friendly
pointer to the right variant and exits.

On macOS the first run of `blueutil` will trigger Privacy & Security
prompts for **Bluetooth** + **Accessibility**. You only need to grant
them once.

## Install

### Option 1 — one-liner from the running cc

The Command Center serves the script at
`/scripts/target_bt_reconnect.sh`. On the target:

```bash
# Replace 192.168.50.251 with the cc host's LAN IP (it's printed
# when you start `handsneyes cc` on the dev Mac).
mkdir -p ~/bin
curl -fsSL http://192.168.50.251:8765/scripts/target_bt_reconnect.sh \
     -o ~/bin/target_bt_reconnect.sh
chmod +x ~/bin/target_bt_reconnect.sh
```

If the target shares the USB-ECM segment with the dev Mac, the
`10.0.0.1:8765` URL works too.

### Option 2 — scp from the repo

```bash
scp scripts/target_bt_reconnect.sh user@target:~/bin/
```

---

## Run

### Diagnostics first (recommended)

If the script "does nothing", run a probe to see exactly what
`bluetoothctl` reports on this host:

```bash
~/bin/target_bt_reconnect.sh --probe
```

It prints the bluez version, adapter state, the parsed paired-
device list, and whether any of `PI_BT_NAMES` matched. Use
`PI_BT_MAC=AA:BB:CC:DD:EE:FF ./target_bt_reconnect.sh --probe`
to override the lookup when the MAC is known.

There's also a one-shot mode that runs a single check and exits
— handy for testing the connect path:

```bash
~/bin/target_bt_reconnect.sh --once
DEBUG=1 ~/bin/target_bt_reconnect.sh --once   # verbose
```

And an explicit re-pair mode that scans for the Pi, pairs, trusts,
and connects in one go — useful when pairing has been lost
entirely:

```bash
~/bin/target_bt_reconnect.sh --pair
SCAN_TIMEOUT=40 ~/bin/target_bt_reconnect.sh --pair    # more patient
```

(The main loop also runs this path automatically whenever no
paired device matches the configured names.)

### Foreground (logs to stdout)

```bash
~/bin/target_bt_reconnect.sh
```

Each line of output is timestamped. Ctrl+C to stop.

### Background — systemd user service (recommended)

Survives logout / reboot. Installs once, runs forever.

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/bt-keepalive.service <<'UNIT'
[Unit]
Description=Keep devmouse BT connection alive
After=bluetooth.service

[Service]
ExecStart=%h/bin/target_bt_reconnect.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now bt-keepalive
```

Tail the log:

```bash
journalctl --user -fu bt-keepalive
```

Stop / disable:

```bash
systemctl --user disable --now bt-keepalive
```

> **Note**: `systemctl --user` services normally stop when you log
> out. To keep them running after logout, enable user lingering:
> `sudo loginctl enable-linger $USER`.

---

## Configuration

Everything is overridable via environment variables — set them in
the `[Service]` block (`Environment="KEY=value"`) or before
launching from a shell.

| var | default | what it does |
|---|---|---|
| `PI_BT_NAMES` | `keyboarder,devmouse` | Comma-separated names to look up. The Pi has been observed under either `keyboarder` (its hostname) or `devmouse` (its BT alias) depending on the pairing flow — the script tries them in order. Override if your Pi has a different name. |
| `PI_BT_MAC` | *(auto-discovered)* | Explicit Pi MAC, e.g. `AA:BB:CC:DD:EE:FF`. Skips the by-name lookup — set this when `--probe` shows no name match. |
| `INTERVAL` | `300` | Seconds between checks. Drop to `60` for snappier recovery during active development. |
| `LOG_FILE` | *(unset)* | Optional path; output is also appended to this file. Useful when running under tmux. |
| `DEBUG` | `0` | Set to `1` to print every `bluetoothctl` response. |

Example with all knobs:

```bash
PI_BT_MAC=A1:B2:C3:D4:E5:F6 INTERVAL=60 LOG_FILE=/tmp/bt-keepalive.log \
    ~/bin/target_bt_reconnect.sh
```

---

## How it decides

Each iteration:

1. Look up the Pi's MAC. Uses `bluetoothctl devices Paired` (bluez
   5.65+) and falls back to `bluetoothctl devices` filtered by
   name on older versions. Re-runs each loop so a freshly-paired
   device gets picked up without a script restart.
2. `bluetoothctl info <mac>` → check for `Connected: yes`. If so,
   continue silently.
3. Otherwise: `power on`, `trust <mac>`, `connect <mac>` (with a
   20s timeout). Logs the outcome:
   - `✓ connected` — success
   - `✓ already connected (no-op)` — `br-connection-already-connected`
   - `✗ connect failed: <tail of bluetoothctl output>` — investigate

The script never **pairs** a new device — only reconnects already-
paired ones. Initial pairing is a one-time manual step (Bluetooth
settings → `devmouse` → Pair).

---

## Troubleshooting

**The script appears to do nothing.**
First, run `./target_bt_reconnect.sh --probe`. If it shows
`no paired device matched`, the script can't find the Pi by
name. Three likely fixes:

  1. The Pi's BT name on this host isn't in the default list.
     `bluetoothctl paired-devices` will show what it is. Re-run
     with `PI_BT_NAMES="that name,keyboarder,devmouse"`.
  2. Pin the MAC: `PI_BT_MAC=AA:BB:CC:DD:EE:FF ./target_bt_reconnect.sh`.
  3. The device genuinely isn't paired here. Open the OS
     Bluetooth settings, pair with the Pi once manually, then
     re-run the script.

**`bluetoothctl: command not found`**
Install bluez: `sudo apt install -y bluez` (Debian/Ubuntu/Kali).

**`device not paired`**
The script is reading bluetoothctl correctly but no paired device
matches any name in `PI_BT_NAMES`. On the target: open Bluetooth
settings, pair with `keyboarder` (or whatever the Pi advertises
as), then check `bluetoothctl paired-devices` shows it. The
script will pick it up automatically on the next loop — no
restart needed.

**`✗ connect failed: br-connection-create-socket`**
The Pi's L2CAP listener isn't open yet. Common right after
`systemctl restart handsneyes-pi`. The script will retry on
the next interval — usually within one cycle.

**`✗ connect failed: br-connection-page-timeout`**
Pi adapter isn't discoverable / powered. On the Pi:
`hciconfig hci0` should show `UP RUNNING PSCAN ISCAN`. If not:
`sudo systemctl restart bluetooth && sudo systemctl restart handsneyes-pi`.

**Reconnects, but the keyboard / mouse still doesn't work**
Pairing is fine but the Pi service isn't serving HID reports.
Hit the Pi's `/health` from the dev Mac:
`curl http://10.0.0.2:8080/health` — `bt_hid_connected: true`
means the Pi's interrupt-channel socket has an active client.
If it stays `false` after a successful `bluetoothctl connect`,
the Pi side needs a restart.
