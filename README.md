# handsneyes

Drive another computer with your voice or text, through a webcam and a Raspberry Pi.

A webcam looks at the target's screen. Multimodal LLMs locate the cursor and whatever you asked it to click. The Pi sends Bluetooth HID keystrokes and mouse moves to the target as if you were sitting there. A closed-loop visual servo refines every click against the webcam image until the cursor lands on the target pixel. Per-OS plugins (`linux_gnome`, `macos`) handle shortcuts and ship the models for that platform's mouse acceleration curve.

## Install

```bash
git clone https://github.com/andrasfe/handsneyes
cd handsneyes
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
brew install tesseract            # macOS — OCR backend
```

## Configure a target

`config/targets.toml`:

```toml
[[target]]
name         = "couch-ubuntu"
platform     = "linux_gnome"
camera_index = 0
pi_url       = "http://10.0.0.2:8080"
transport    = "bt"
screen_size  = [1920, 1080]
```

`camera_index` accepts an integer or the string `"auto"`. With
`"auto"`, the loader probes cv2 indices, vetoes any device that's
just mirroring the dev mac's own desktop (useless self-capture), and
prefers low-temporal-noise feeds (a screen-share virtual camera
carrying the remote machine's mirrored desktop) over high-noise ones
(a physical webcam pointed at a monitor). The choice is cached
process-wide, so several `"auto"` targets share one probe pass.

### Self-driving the same Mac

Set `capture_source = "screen"` and add `platform = "macos"` to drive
the same Mac the cc runs on. The Command Center attaches a Quartz-
based cursor oracle (`CGEventGetLocation`) so the homer reads the
cursor directly without trying to find it in the captured frame —
macOS doesn't composite the cursor into the framebuffer that
`screencapture` / PIL `ImageGrab` see, so vision-only loops never
converge on self-capture.

### Controlling a remote Mac via screen-share

Mirror the target Mac to the dev mac via AirPlay or Sidecar, exposing
its desktop as a virtual camera. Then add:

```toml
[[target]]
name         = "remote-mac"
platform     = "macos"
camera_index = "auto"                  # picks the screen-share device
pi_url       = "http://10.0.0.2:8080"
transport    = "bt"
screen_size  = [1920, 1080]            # whatever the virtual camera streams at
```

Three setup notes:

1. **Pi BT HID pairing.** Bluetooth HID is point-to-point — the Pi
   can hold one host at a time. On the remote Mac, *Settings →
   Bluetooth → connect* to the Pi (e.g. "TerminalEyes HID" or
   "keyboarder"). If another Mac is still paired, *Forget* the Pi on
   that Mac first.
2. **Effective resolution gotcha.** The pointer-accel model is
   trained at one *effective* ("UI Looks like") display resolution.
   Different effective resolutions on dev and target produce a
   constant percent-keyed over- or under-shoot. The cc UI has an
   `accel X × Y` control in the chat row — set it to
   `(target_effective_w / dev_effective_w, target_effective_h /
   dev_effective_h)`. Example: dev at *Looks like 3840 × 2160* and
   remote at *Looks like 1728 × 1117* → `0.45 × 0.52`. Find the
   numbers via `system_profiler SPDisplaysDataType | grep -E
   "Resolution|UI Looks like"`.
3. **Default macOS cursor is hard to track.** The HSV finder was
   tuned against Ubuntu's `redglass`; on macOS, enable a high-contrast
   pointer (Settings → Accessibility → Display → Pointer: max
   saturation, max size) so the homer's CV cursor finder can lock on.

## Run

**Command-line:**

```bash
handsneyes do --target couch-ubuntu "click the Firefox icon"
handsneyes do --target couch-ubuntu "go to reddit.com"
handsneyes do --target couch-ubuntu "scroll down 5"
handsneyes do --dry-run "scroll down 5"        # plan-only, no HID
```

**Web UI (Command Center):**

```bash
handsneyes cc                                  # http://localhost:8765
```

Click on the live webcam feed to move the host cursor; type into the passthrough field to send keys; press the Unlock button for a guided login flow with three tabs (use a vault entry, type the password once, or create a fresh vault).

**On the Pi:**

```bash
handsneyes-pi                                  # FastAPI HID gateway on :8080
```

## Add a new OS

1. `cp -r src/handsneyes/platforms/headless src/handsneyes/platforms/<os>` and rename the class.
2. Implement `open_app`, `focus_browser`, `window_action`, `remap_combo` in the new adapter.
3. Train weights with `scripts/collect_pointer_accel.sh` → `scripts/build_pointer_accel_dataset.py` → `scripts/train_pointer_accel.py`. Drop the resulting `config.json` + `weights.npz` into `platforms/<os>/models/pointer_accel/`.
4. Register in `pyproject.toml`:
   ```toml
   [project.entry-points."handsneyes.platforms"]
   <os> = "handsneyes.platforms.<os>:Adapter"
   ```
5. Add a target row in `targets.toml` with `platform = "<os>"`.

## Project layout

```
src/handsneyes/
├── core/         # OS-agnostic agents + vision
├── io/           # keyboard/mouse abstractions
├── platforms/    # per-OS plugins (linux_gnome, macos, headless)
├── targets/      # multi-host registry
├── ui/           # FastAPI Command Center
├── pi/           # rsync this to the Raspberry Pi
└── cli.py
```

Deep dive — agent architecture, ML training pipeline, gotchas, every operational lesson learned: [`CLAUDE.md`](CLAUDE.md).
