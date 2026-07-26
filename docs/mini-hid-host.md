# Mini as the HID host (no Raspberry Pi)

A regular Linux desktop/mini can *be* the Bluetooth mouse itself, instead of
using a separate Raspberry Pi. Its own Bluetooth radio emulates an HID
mouse/keyboard ("devmouse") that the target Mac pairs with — no Pi, no
USB-ECM link, no Pi-side bond management.

Use this when the mini is within Bluetooth range (~10 m) of the target.

## Why you might prefer it

- Eliminates the Pi as a failure domain (USB link drops, Pi reboots, IP loss).
- Bond/pairing state lives on the mini, where you have full local access —
  stale-bond problems are fixed with one local command, no Pi password.

## Trade-offs

- The mini must stay within BT range of the target.
- The mini's Bluetooth becomes dedicated to this (audio profiles are stripped;
  LE is disabled). Don't expect to use it for headphones meanwhile.
- It does **not** change macOS's reluctance to accept an emulator's
  *self-initiated* reconnect after sleep — the same as any BT-HID emulator.
  What it *does* change is that recovery is now local and instant.

## Setup

Run on the **mini** (the Linux host that will emulate the mouse).

### 1. Build the gateway venv (normal user)

```bash
bash scripts/setup_mini_gateway_venv.sh
```

Creates `.venv-gw/` holding `afferent-gateway` plus its deps. It uses
`--system-site-packages` (to reach the system `python-dbus`) but installs the
web stack with `PYTHONNOUSERSITE=1` so nothing is borrowed from your
`~/.local` — the gateway later runs as **root**, which can't see your
user-site.

### 2. Configure BlueZ + launch the gateway (root)

```bash
sudo bash scripts/setup_mini_hid.sh
```

This reconfigures `bluetoothd` (`--compat`, input+audio plugins disabled),
sets the adapter to advertise as **devmouse** with the HID device class,
strips the audio SDP records, starts the auto-accept pairing agent, and
launches the gateway on `http://localhost:8080`. Logs: `/tmp/mini-gateway.log`,
`/tmp/mini-bt-agent.log`.

Verify:

```bash
curl -s http://localhost:8080/health   # bt_hid_connected:false until the Mac pairs
```

### 3. Pair the target Mac

On the Mac: **System Settings → Bluetooth**, pair the new **devmouse** device.
No stale bond exists, so it pairs cleanly. Confirm on the mini:

```bash
curl -s http://localhost:8080/health   # -> bt_hid_connected:true, bt_hosts:[...]
```

Quick motion test (cursor should sweep on the Mac):

```bash
.venv-gw/bin/python - <<'PY'
from afferent import GatewayClient; import time
g = GatewayClient("http://localhost:8080")
g.move_large(400, 0); time.sleep(0.3); g.move_large(-400, 0)
PY
```

### 4. Point the Command Center at the local gateway

```bash
cp config/targets.example.toml config/targets.toml
```

Keep the `mac-via-mini` target (its `pi_url = "http://localhost:8080"`); set
`bt_host_mac` to the Mac's BT MAC if the gateway holds more than one host.
Restart `handsneyes cc`.

## Troubleshooting

- **`ModuleNotFoundError` when the gateway starts as root** — a dep resolved
  from `~/.local`. Re-run `setup_mini_gateway_venv.sh`; it forces deps into the
  venv with `PYTHONNOUSERSITE=1`.
- **Mac still shows the old adapter name, not "devmouse"** — macOS caches the
  BT name per-MAC. Toggle the Mac's Bluetooth off/on to flush.
- **Mac audio routes to the mini after pairing** — the audio SDP strip didn't
  run; run `sudo bash scripts/bt-strip-audio-sdp.sh`, then reselect your normal
  output on the Mac once.
- **Cursor doesn't move but moves return `ok`** — HID reports are accepted but
  not landing as motion; re-pair from the Mac (Forget → pair again).

## Reverting to a normal Bluetooth stack

```bash
sudo cp /etc/bluetooth/main.conf.pre-mini-hid /etc/bluetooth/main.conf
sudo rm /etc/systemd/system/bluetooth.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart bluetooth
```
