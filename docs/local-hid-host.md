# Local host as the HID device (no Raspberry Pi)

A regular Linux machine can *be* the Bluetooth mouse itself, instead of using a
separate Raspberry Pi. Its own Bluetooth radio emulates an HID mouse/keyboard
("devmouse") that the target Mac pairs with — no Pi, no USB-ECM link, no
Pi-side bond management.

Use this when the machine running handsneyes is within Bluetooth range (~10 m)
of the target.

## Why you might prefer it

- Eliminates the Pi as a failure domain (USB link drops, Pi reboots, IP loss).
- Bond/pairing state lives on the local host, where you have full access —
  stale-bond problems are fixed with one local command, no Pi password.

## Trade-offs

- The host must stay within BT range of the target.
- The host's Bluetooth becomes dedicated to this (audio profiles are stripped;
  LE is disabled). Don't expect to use it for headphones meanwhile.
- It does **not** change macOS's reluctance to accept an emulator's
  *self-initiated* reconnect after sleep — the same as any BT-HID emulator.
  What it *does* change is that recovery is now local and instant.

## Setup

Run on the **local host** (the Linux machine that will emulate the mouse).

### 1. Build the gateway venv (normal user)

```bash
bash scripts/setup_local_gateway_venv.sh
```

Creates `.venv-gw/` holding `afferent-gateway` plus its deps. It uses
`--system-site-packages` (to reach the system `python-dbus`) but installs the
web stack with `PYTHONNOUSERSITE=1` so nothing is borrowed from your
`~/.local` — the gateway later runs as **root**, which can't see your
user-site.

### 2. Configure BlueZ + launch the gateway (root)

```bash
sudo bash scripts/setup_local_hid.sh
```

This reconfigures `bluetoothd` (`--compat`, input+audio plugins disabled),
sets the adapter to advertise as **devmouse** with the HID device class,
strips the audio SDP records, starts the auto-accept pairing agent, and
launches the gateway on `http://localhost:8080`. Logs: `/tmp/hid-gateway.log`,
`/tmp/hid-agent.log`.

Verify:

```bash
curl -s http://localhost:8080/health   # bt_hid_connected:false until the Mac pairs
```

### 3. Pair the target Mac

On the Mac: **System Settings → Bluetooth**, pair the new **devmouse** device.
No stale bond exists, so it pairs cleanly. Confirm on the host:

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

Keep the local-gateway target (its `pi_url = "http://localhost:8080"`); set
`bt_host_mac` to the Mac's BT MAC if the gateway holds more than one host.
Restart `handsneyes cc`.

## Surviving reboots

`setup_local_hid.sh` launches the agent and gateway as ad-hoc background root
processes — they die on reboot and the mouse silently stops. Install them as
systemd services instead (once):

```bash
sudo bash scripts/install_local_hid_services.sh      # --uninstall to remove
systemctl status handsneyes-gateway handsneyes-btagent
journalctl -u handsneyes-gateway -f
```

Both carry `Restart=always` and start after `bluetooth.service`, re-asserting
the adapter's runtime state (alias, and pairable/discoverable only while
nothing is bonded) before the gateway binds L2CAP.

## Troubleshooting

- **`ModuleNotFoundError` when the gateway starts as root** — a dep resolved
  from `~/.local`. Re-run `setup_local_gateway_venv.sh`; it forces deps into the
  venv with `PYTHONNOUSERSITE=1`.
- **Mac still shows the old adapter name, not "devmouse"** — macOS caches the
  BT name per-MAC. Toggle the Mac's Bluetooth off/on to flush.
- **Mac audio routes to this host after pairing** — the audio SDP strip didn't
  run; run `sudo bash scripts/bt-strip-audio-sdp.sh`, then reselect your normal
  output on the Mac once.
- **Other machines keep popping up "devmouse" pairing prompts** — the adapter
  is still advertising. Once the target is bonded you don't need it:
  `bluetoothctl discoverable off && bluetoothctl pairable off`. A bonded target
  reconnects without either.
- **The target won't connect at all** (no attempt even reaches us — the adapter
  never shows `Connected: yes`): the *local* side is holding a stale bond that
  the target no longer matches, and BlueZ rejects its attempts at the radio
  level, silently, with nothing in any log. Clear our bond and let it pair
  fresh — this is the single most common "it just won't connect":
  ```bash
  bluetoothctl remove <TARGET_MAC>
  bluetoothctl pairable on && bluetoothctl discoverable on
  # then pair "devmouse" again from the target; re-trust afterwards:
  bluetoothctl trust <TARGET_MAC>
  bluetoothctl pairable off && bluetoothctl discoverable off
  ```
  A fresh pairing resets `Trusted`, so set it again or the gateway's reconnect
  watchdog logs `no trusted devices, sleeping` and never helps.
- **Cursor doesn't move but every call returns `ok`** — the most common failure,
  and the status is misleading: `bt_hid_connected: true` plus `200 OK` only mean
  the report was written to an open socket, not that macOS consumed it. The
  session can go stale (socket open, host ignoring it) with no error anywhere.
  Fix: disconnect and reconnect **from the Mac**. Note that connecting from this
  side (`bluetoothctl connect`) brings up the ACL link but *not* the HID
  channels — only the host can open those, so the reconnect must originate on
  the Mac.

## Reverting to a normal Bluetooth stack

```bash
sudo cp /etc/bluetooth/main.conf.pre-handsneyes-hid /etc/bluetooth/main.conf
sudo rm /etc/systemd/system/bluetooth.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart bluetooth
```
