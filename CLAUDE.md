# CLAUDE.md

## Project

**handsneyes** — terminaleyes 2.0. Vision-based agentic terminal controller, restructured around per-OS plugins, multi-host target registry, and a clean tiered agent layer. A webcam captures the target's screen; classical CV + multimodal LLMs locate the cursor and the click target; HID commands flow over BT (or USB) via a Raspberry Pi to drive the target machine. Target OS may be Linux/GNOME (primary, model-tuned) or macOS (adapter shipped, no model yet).

Lives at `github.com/andrasfe/handsneyes`. Migrated from `terminaleyes` repo via the 3-phase architecture plan in `~/.claude/plans/splendid-gathering-moore.md`. Migration is complete; this repo is the active codebase.

## Top-level layering rule (load-bearing)

- **`core/`** depends only on `io/` interfaces + the active `PlatformAdapter` injected via `AgentContext`. Never imports from `platforms/<anything>` or `pi/`.
- **`io/`** depends on stdlib + httpx + cv2 only. No OS-specific code.
- **`platforms/<os>/`** imports core types (ABC, dataclasses) and registers via entry_points. May NOT import from other platforms.
- **`ui/`** depends on `core/` + `targets/`. Knows the active target's `platform_name` string, not the adapter class.
- **`pi/`** is its own deployable; never imports from `src/handsneyes/`.

This layering invariant is what makes adding a new OS a strictly additive change.

## Agent architecture

Tiered agents under `src/handsneyes/core/agents/`:

- **Tier 1 (atomic)** — `VerifyAgent` (visual yes/no via multimodal LLM), `CursorAgent` (HSV → oscillation-variance cascade), `TargetAgent` (OCR → ShowUI grounding)
- **Tier 2 (actions)** — `WakeAgent` (jiggle/arrow/click + brightness self-check), `TypeAgent` (text input with `secret=True`), `ScrollAgent` (mouse-wheel scroll, optional hover-at)
- **Tier 3 (workflows)** — `FocusAgent` (centre app via adapter `window_action("maximize")`), `LoginAgent` (wake+verify+type with vault or direct password), `NavigateAgent` (browser-aware URL bar typing with post-OCR oracle), `ClickAgent` (wraps `VisualServoHomer` with scroll-and-retry)
- **Tier 4 (storage)** — `Vault` (AES-256-GCM file with scrypt KDF, reads both `HEVAULT1` + legacy `TEVAULT1` magic)
- **Controller** — `ControllerAgent` decomposes free-form intents via a rule planner; threads `vault_name`/`password`/`skip_verify` from the cc options into login PlanSteps. Wraps `PlanExecutor` which captures pre/post snapshots between steps.

Each agent returns a typed `Outcome { success: bool, reason: str, data: dict }`. Higher-tier agents construct lower-tier ones with the same `AgentContext` so I/O resources (capture, mouse, keyboard, vision client, vault, platform adapter, output dir) are wired once per session.

Defaults that make the controller "safe":
- Click-like intents are prefixed with `FocusAgent` unless `--no-focus`.
- `NavigateAgent` refuses to send keystrokes until `VerifyAgent` confirms a browser is the foreground app — falls back to `ctx.platform.focus_browser()` which on linux_gnome runs Super → activities → type "firefox" → Enter → re-verify.
- `LoginAgent` refuses to type when `VerifyAgent` doesn't see a login screen — visual-only judgement, NOT keyword matching for "password". Override via `verify=False` (cc UI exposes "Skip visual verify").
- `FocusAgent` refuses to act on dark/asleep frames (brightness check).

Full index in `AGENTS.md`.

## The PlatformAdapter abstraction (the linchpin)

`platforms/base.py::PlatformAdapter` is the contract every per-OS plugin implements:

```python
class PlatformAdapter(ABC):
    name: str                                                   # "linux_gnome"
    display_name: str                                           # "Linux / GNOME"
    package_root: Path                                          # for models/

    def capabilities(self) -> Capabilities                      # feature flags
    def canonicalise_app(self, alias: str) -> AppHint           # "shell" → terminal
    async def open_app(self, kb, *, app, settle_ms) -> None
    async def focus_browser(self, ctx, *, attempt, max_attempts) -> str
    async def window_action(self, kb, intent: WindowIntent) -> None
    def remap_combo(self, mods, key) -> tuple[list[str], str]   # Ctrl→Cmd on mac

    # Optional — default None / no-op
    def cursor_theme_advice(self) -> CursorThemeAdvice | None
    def pointer_accel_checkpoint(self) -> Path | None
    def longjump_checkpoint(self) -> Path | None
    def login_hint(self) -> LoginHint | None
```

Injected into `AgentContext.platform`. Shortcut remap happens at the I/O boundary via `PlatformKeyboard(raw_kb, adapter)` proxy — so `send_key_combo(["ctrl"], "a")` automatically becomes Cmd+A on macOS without agents knowing.

## Plugin registration

`pyproject.toml`:
```toml
[project.entry-points."handsneyes.platforms"]
headless    = "handsneyes.platforms.headless:HeadlessAdapter"
linux_gnome = "handsneyes.platforms.linux_gnome:LinuxGnomeAdapter"
macos       = "handsneyes.platforms.macos:MacOSAdapter"
```

`platforms/__init__.py::load_adapter(name)` resolves via `importlib.metadata.entry_points`. `$HANDSNEYES_PLATFORM` is a debug override.

## Multi-host target registry

`config/targets.toml` (priority: `$HANDSNEYES_TARGETS_FILE` > `./config/targets.toml` > `~/.config/handsneyes/targets.toml` > built-in headless):

```toml
[[target]]
name         = "couch-ubuntu"
platform     = "linux_gnome"
camera_index = 0
pi_url       = "http://10.0.0.2:8080"
transport    = "bt"
screen_size  = [1920, 1080]
```

`handsneyes do --target couch-ubuntu "..."` resolves the name → loads the adapter → constructs an `AgentContext` from the target's HID + camera. The cc auto-picks the first non-headless target at boot (single-target for now; multi-target dropdown in the UI is a follow-up).

## Per-OS lightweight models

Models ship inside the platform package:
- `src/handsneyes/platforms/linux_gnome/models/pointer_accel/{config.json,weights.npz}`
- `src/handsneyes/platforms/linux_gnome/models/longjump/{config.json,weights.npz}`

Adapter exposes `pointer_accel_checkpoint()` / `longjump_checkpoint()`. The `VisualServoHomer` searches the platform-bundled paths first, then falls back to the legacy `data/ml/checkpoints/*` locations.

**Current shipped models (Yaru white cursor):**
- `pointer_accel-yaru-v6` (hidden=64, 4083 training rows, val_mse 0.032, **canary median 9.5, mean 10.0, range 5-17**)
- `longjump-yaru-v1` (25-trajectory dataset, marginal quality; runtime falls back to closed-loop seed when it misfires)

**Retrain history (Yaru):**
- v4: 2333 rows, click_at-collected. Original baseline. Median 8.5 on its first canary; drifted to ~10 over weeks of use.
- v5: +500 active-learning rows (`explore_pointer_accel_v2.py`). Regressed (canary median 9.5 → 10.5). Rolled back.
- v6: +1000 more active-learning rows on top of v5's corpus → 4083 train / 505 val / 488 test. Beat the same-session v4 baseline on every metric (median 9.5 vs 10.0, mean 10.0 vs 11.25, min 5 vs 8). Shipped.

Lesson from v5 → v6: small active-learning batches (500 rows) are too dilute against a 2333-row baseline corpus to move the model; need ≥1000 to make the new distribution dominant. The val_mse number gets *worse* as the active-learning rows mix in (they're inherently harder samples), but the canary is the only ground truth — don't reject a retrain on val_mse alone.

## ML training pipeline

The pointer_accel model maps `(target_dx_pct, target_dy_pct, cursor_x_pct, cursor_y_pct) → (hid_dx, hid_dy)`. It's the homer's open-loop *first-iteration seed*: given a target pixel delta + current cursor position, predict how many HID units to send so the cursor lands close. The closed-loop visual servo then refines from there. A good model means fewer servo iterations per click (median steps drop ~3x from ratio-only seed → trained model).

The longjump model is similar but predicts the *total* HID for a slam-to-target (chained-burst pattern, no captures between bursts). Cursor-independent in principle since it's about HID→pixel dynamics, not visual detection.

### Full retraining loop

The pipeline runs against the live cc on the live target. Every step builds on the previous artefact.

**1. Baseline.** Fire 8 canary clicks at fixed positions (corners + edges + centre), record `steps` from each `click_at` response:
```bash
for xy in "0.15 0.15" "0.5 0.15" "0.85 0.15" "0.15 0.50" "0.85 0.50" \
          "0.15 0.85" "0.5 0.85" "0.85 0.85"; do ...; done
```
Record median, mean, max. Identify weak regions (consistently >12 steps → bad coverage or undertrained).

**2. Collect.** Each `click_at` runs the homer end-to-end and persists every servo step to `<run>/homer/<id>/history.jsonl` (cursor position before, sent HID, measured delta, ratio, note). Three patterns we use:
- **Uniform grid** (`scripts/collect_pointer_accel.sh --grid 10` → 10×7=70 points): broad coverage.
- **Offset grid** (`--grid 12` → 12×9=108): different start positions, fills interpolation gaps.
- **Targeted oversample** (`scripts/collect_targeted.sh`): hand-picked positions in regions baseline showed weak. Currently the left strip (x ∈ [0.05, 0.25]) and bottom strip (y ∈ [0.78, 0.94]).

Each click ≈ 6s wall time. 232 clicks (the v4 corpus) takes ~25 minutes.

**3. Build dataset.** `scripts/build_pointer_accel_dataset.py --since <epoch>` walks the runs dir, filters out:
- Files older than `--since` (essential after a cursor theme change — the redglass-cursor history is wrong for white-cursor training).
- Rows with `hid_dx==0 && hid_dy==0` (post-click confirm records, not pointer-accel samples).
- Rows missing `measured_dx_pct`/`measured_dy_pct`.
- Rows outside the sane pct-per-hid ratio window `[3e-4, 8e-3]` (operator moved the mouse mid-click, modal popped up, etc. — would poison training).

Splits 80/10/10 by trajectory (so train/val/test never share trajectories — prevents leakage). Output: `data/ml/pointer_accel/{train,val,test}.jsonl`.

**4. Train.** `scripts/train_pointer_accel.py --hidden 64 --epochs 400 --lr 5e-3`. Architecture: 2-layer MLP with GELU activations. Direction `inverse` (default in v3+) — input is `(measured_dx_pct, measured_dy_pct, cursor_centred)`, output is `(hid_dx_norm, hid_dy_norm)`. Runtime then does a single forward pass — no Newton iteration needed.

Trainer hardenings (added during the v4 retrain):
- **Best-val checkpointing**: tracks `val_mse` every epoch, saves the weights at the lowest. Prevents the "trained 400 epochs but kept the last-epoch overfit weights" trap.
- **Augmentation**: sign-flipping along both axes (cheap 4× data via symmetry).
- **Per-epoch validation**: cheap on this dataset size (~200 val rows × forward pass = ms).

**5. Install.** Copy `data/ml/checkpoints/pointer_accel-yaru-v<N>/{weights.npz,config.json}` into `src/handsneyes/platforms/linux_gnome/models/pointer_accel/`. Restart cc so the homer reloads the model on next click_at.

**6. Canary.** Same 8 points as the baseline. Compare median/mean/max steps. Then *also* test 8 fresh points to confirm the model generalises, not just fits the baseline-grid neighbours.

### Hyperparameter lessons learned (Yaru retrain campaign)

Three variants trained on the same 2333-row corpus:

| variant | hidden | lr | best epoch | val_mse | canary median | canary max |
|---|---|---|---|---|---|---|
| v3 | 128 | 1e-2 | 398 | 0.009 | 11 | **30** (outlier) |
| **v4** | **64** | **5e-3** | **99** | **0.0096** | **8.5** | **13** |
| v2 (pre-campaign, 478 rows) | 48 | 1e-2 | 200 | 0.058 | 13 | 17 |

Key findings:
- **More data dominates architecture choices.** v2 → v4 with same-ish architecture but 5× data cut median steps 35%.
- **Hidden=128 overfit despite early-stop.** The 30-step outlier was a regression — model fit noise in some neighbourhood. Hidden=64 generalises better at this data scale.
- **lr=5e-3 with best-val checkpointing finds the sweet spot at epoch ~100.** Higher lr (1e-2) trains faster but the best-val is reached later and noisier.
- **Augmentation is essential.** Without sign-flipping the trainer overfits direction-specific quirks (e.g. left-going moves get a different ratio than right-going on libinput-adaptive, but the *shape* of the curve is symmetric).

### Dataset growth path

| campaign | trajectories | rows | model | median steps |
|---|---|---|---|---|
| Redglass cursor (terminaleyes legacy) | ~600 | ~3000 | pointer_accel-v5 | ~3 (HSV fast-path) |
| White cursor v1 (initial Yaru port) | 25 | 176 | yaru-v1 | ~10 |
| White cursor v2 | 65 | 478 | yaru-v2 | 13 |
| **White cursor v4 (current)** | **328** | **2333** | **yaru-v4** | **8.5** |

Asymptote: oscillation-variance + frame-diff detection on a white cursor is fundamentally ~3-5 steps slower per servo iteration than HSV on the red cursor (more captures needed per measurement). Realistic best-case for this configuration is probably 5-7 steps median; below that needs either the redglass theme back, or a vision model that can locate the cursor in a single frame.

### Host-side data collection (the noise-free path)

The webcam-mediated explorers on the dev mac (`explore_pointer_accel.py`, `explore_pointer_accel_v2.py`) have a ±3-5 px label-noise floor from oscillation-variance cursor detection on Yaru white. That floor has bottlenecked every retrain (see the v5 attempt: 500 active-learning rows on the dev mac took the canary from 9.5 → 10.5 median).

`scripts/host_collect.py` is the way around it. It runs ON THE TARGET MACHINE, injects relative mouse deltas through `/dev/uinput` (so libinput applies the same acceleration curve as the Bluetooth HID path), and reads cursor pixel positions via `xdotool getmouselocation`. Both sides of the (HID, pixel) pair are pixel-exact — no webcam, no detector noise.

Throughput: ~1000-2000 rows/min (vs 30 with the webcam path). A 10-min host session produces more trainable rows than every previous retrain campaign combined.

Deployment (one-off, on the target):
```
sudo apt install xdotool python3-evdev
sudo usermod -aG input $USER   # log out + back in
scp scripts/host_collect.py <host>:~/
ssh <host> python3 host_collect.py --samples 5000 --out ~/host_collect.jsonl
scp <host>:~/host_collect.jsonl \
    ~/.local/share/handsneyes/runs/host_collect_$(date +%Y%m%d)/homer/host-001/history.jsonl
```

Then the existing `build_pointer_accel_dataset.py` + `train_pointer_accel.py` work unchanged — the row schema matches.

When pairing the dev-mac Claude with a host-side Claude: the host instance just runs `host_collect.py` (or modifies sampling strategy) and `scp`s the file back. The dev-mac instance handles dataset assembly, training, canary, install. Two-machine handoff, neither has to do the other's job.

### When to retrain

- **Cursor theme changed.** Critical — the pct-per-hid distribution shifts because detection latency changes. Always use `--since <epoch>` to filter out pre-change history.
- **Target OS changed** (Ubuntu/libinput → macOS/IOHID for example). The acceleration curve is OS-specific.
- **>500 new trajectories accumulated** since last train. Marginal returns flatten after ~2000 trajectories on this architecture.
- **Canary median creeps up by >20%** over a week of normal use. Something drifted — webcam angle, target screen DPI, lighting. Retrain on recent data.

### Longjump status

Longjump v3 retrain attempt (350 trajectories) ended with val_mse 0.6 + huge train/val gap = bad overfit. Not shipped. Theory: the trajectories the homer collects don't have enough variety in the *target* dimension — most clicks slam from (0,0) to somewhere, not from (0.4, 0.6) to (0.7, 0.3). The longjump dataset is over-concentrated on one slice of input space. Future work: collect trajectories starting from random cursor positions (need to add a "random pre-position" step to `collect_pointer_accel.sh`).

The existing redglass-trained longjump is still shipped; cursor-independent dynamics mean it still gives the homer a usable first-burst HID seed. When it misfires (HID overshoot/undershoot >25%), the closed-loop servo absorbs the residual.

## Pi-side architecture

Same as terminaleyes — `src/handsneyes/pi/` ports verbatim with brand-string renames:
```
[Dev Mac / Agent] --USB ECM Ethernet--> [Pi Zero 2 W] --BT HID--> [Target Mac/Linux]
     10.0.0.1                              10.0.0.2
```

`handsneyes-pi` entry-point binary runs on the Pi. SDP record advertises as "handsneyes HID". `bluetoothd --noplugin=input --compat` is mandatory (see /docs/pi-setup or the original terminaleyes CLAUDE.md for the full debugging checklist — those hard-won lessons still apply).

## Session output dir

Every captured frame goes to a single per-invocation directory (replayable). Resolution order:
1. `--output-dir PATH` CLI flag
2. `HANDSNEYES_OUTPUT_DIR` env var
3. `~/.local/share/handsneyes/runs/`

Cc per-run subdir: `<watch_dir>/<run_id>/`. Manual captures (click_at, snapshot, sync) write to `<watch_dir>/manual/`. The `FrameStore` polls the watch_dir for new images and streams them via SSE to the UI.

## Commands

```bash
pip install -e ".[dev]"                    # install
brew install tesseract                     # OCR backend
pip install pytesseract mlx                # python bindings + training backend
python -m pytest tests/ -v                 # run tests (213 currently passing)

# Common flags
handsneyes --output-dir PATH ...
HANDSNEYES_TARGETS_FILE=PATH handsneyes ...

# Controller
handsneyes do "click the Run button" --target couch-ubuntu
handsneyes do "go to reddit.com/r/LocalLLaMA"
handsneyes do "scroll down 6"
handsneyes do --dry-run "<intent>"

# Direct agent invocations
handsneyes targets                         # list configured targets
handsneyes platforms                       # list registered adapters
handsneyes vault {add,get,list,remove,status}

# Command Center
handsneyes commandcenter                   # http://0.0.0.0:8765
handsneyes cc                              # alias
handsneyes cc --frames-dir PATH

# Pi side (on the Pi only)
handsneyes-pi                              # FastAPI on :8080
```

## Command Center (`ui/`)

FastAPI app at `http://localhost:8765`. Wiring at startup:
- **FrameStore** polls watch dir every 250ms, indexes new PNGs, exposes via SSE.
- **LogBus** captures the `handsneyes` logger + redirects active-run stdout/stderr.
- **Runner** is one-at-a-time. Each `/api/run` builds a fresh `AgentContext` via `make_target_context_factory(target, adapter, base_dir, bus)`. Closes mouse/keyboard/capture when run ends — webcam is only held during a run.
- **Per-request overrides**: `RunRequest` accepts `password`, `vault_passphrase`, `skip_verify`. The runner builds a per-request Vault when `vault_passphrase` is set, threads everything into the login PlanStep.

Endpoints (selection):
| Method | Path | Notes |
|--------|------|-------|
| POST | `/api/run` | Body: RunRequest with intent + options. Password and vault_passphrase redacted from public records. |
| POST | `/api/mouse/click_at` | Body: `{x_pct, y_pct, button, count}`. Drives VisualServoHomer; ~8-12 steps with current model. |
| POST | `/api/snapshot` | Single fresh webcam frame. `?dedup=1` skips persistence if unchanged. |
| POST | `/api/sync-text-from-host` | OCRs the focused host text field via `nanonets-ocr-s`. Optional `{x_pct, y_pct, band_pct}` ROI; defaults to last click_at position. |
| POST | `/api/keyboard/key`, `/text` | Direct HID. Bypass controller. |
| POST | `/api/vault/create` | Body: `{passphrase, entry_name, entry_value, overwrite}`. Creates fresh vault file. |
| GET | `/api/runs[/{id}[/logs]]` | SSE log stream replays from buffer. |

### Unlock modal (UI)

Click Unlock → modal with three tabs:
1. **via vault** — entry name + master passphrase. Cached in JS heap for the session (wiped on any failed run).
2. **type directly** — bypass vault, type the password inline (never persisted).
3. **create new vault** — fresh passphrase + confirm + entry name + value. Overwrites any existing vault file. Useful when the master passphrase is forgotten.

Plus a **"Skip visual verify"** checkbox (session-sticky) for when the webcam is fooled (e.g. SMPTE bars from a busy/disconnected camera). All password fields are `type="password"` and wiped on modal close.

### Passthrough field (host typing)

Below the screenshot. Typing into it or hovering the screenshot forwards every keystroke to the host. The local mirror tracks caret + selection so Left/Right/Home/End/Backspace/Delete work as expected, type-mid-string inserts at caret, etc.

**📥 Sync button**: OCRs the host's focused text field via `nanonets-ocr-s` and replaces the local mirror's value with what's actually on the host. Auto-fires (silent, 250ms delay) after every click_at — so clicking into a pre-populated host text field automatically pulls the content into the local mirror.

## Vault

AES-256-GCM file with scrypt KDF. Magic bytes:
- New writes: `HEVAULT1` at `~/.config/handsneyes/vault.enc`
- Legacy reads: `TEVAULT1` at `~/.config/terminaleyes/vault.enc` (transparent fallback so terminaleyes migrants don't lose entries)

Passphrase priority: `HANDSNEYES_VAULT_PASSPHRASE` env var > `TERMINALEYES_VAULT_PASSPHRASE` (legacy compat) > inline (from cc UI's `vault_passphrase` field) > interactive getpass (CLI only). Empty strings treated as "not set."

File mode `0600`, dir mode `0700`. Atomic write via tempfile rename.

## Known gotchas

- **Camera index**: targets.toml uses `camera_index = 0` (Guermok USB3). If macOS returns SMPTE colour bars from `cv2.VideoCapture(0)`, the camera is either unplugged, held by another app (Photo Booth / Slack / Zoom), or TCC camera permission for the Terminal/Python process was revoked. cc can't recover this — operator must fix at the macOS layer.
- **Vision model identifier**: LM Studio registers nemotron as `nvidia/nemotron-3-nano-omni` (qualified). The bare `nemotron-3-nano-omni` 400s with `model_not_found`. `ui/factory.py` uses the qualified ID.
- **Cursor theme**: target machine's cursor is the Yaru default white. `LinuxGnomeAdapter.cursor_theme_advice()` documents the optional redglass theme for HSV fast-path, but the homer works fine on Yaru via oscillation-variance + frame-diff.
- **Webcam open hangs**: `cv2.VideoCapture(N)` on macOS can block forever when N points at a busy / phantom device. The cc factory wraps `capture.open()` in `asyncio.wait_for(..., 10.0)` so a hung device can't peg the runner.
- **Pi BT HID**: `bluetoothctl connect` from the Pi will crash WiFi on the BCM43436s (shared radio). With ECM mode WiFi is freed for BT.
- **Cached vault passphrase trap**: a wrong-passphrase attempt used to silently feed every subsequent Unlock click. Modal now always opens; cache is wiped on any failed unlock-intent run.
- **Pre-cursor-switch training data**: when retraining pointer_accel, always pass `--since <epoch>` to `build_pointer_accel_dataset.py` to filter out history collected under a different cursor theme.

## Key directories

```
handsneyes/
├── pyproject.toml                # entry_points + deps
├── config/targets.toml           # multi-host target registry
├── data/ml/                      # local-only retraining artefacts (gitignored)
├── scripts/
│   ├── collect_pointer_accel.sh  # grid click_at → history.jsonl rows
│   ├── collect_targeted.sh       # oversample model-weak regions
│   ├── build_pointer_accel_dataset.py
│   ├── train_pointer_accel.py    # MLX MLP, best-val checkpoint
│   └── build/train_longjump.py
└── src/handsneyes/
    ├── core/
    │   ├── agents/               # base, context, verify, cursor, target,
    │   │                         # wake, type, scroll, focus, login,
    │   │                         # navigate, click, controller, executor
    │   ├── vision/               # cursor_finder, ocr_finder,
    │   │                         # visual_servo_homer, closed_loop_homer,
    │   │                         # pointer_accel + longjump loaders,
    │   │                         # session_adapter (homer bridge)
    │   ├── capture/              # WebcamCapture (cv2)
    │   └── vault/                # AES-GCM file vault
    ├── io/
    │   ├── keyboard/             # base ABC, HttpKeyboardOutput,
    │   │                         # PlatformKeyboard proxy (remap_combo)
    │   └── mouse/                # base ABC, HttpMouseOutput
    ├── platforms/
    │   ├── base.py               # PlatformAdapter ABC + dataclasses
    │   ├── headless/             # identity adapter for tests + dry-run
    │   ├── linux_gnome/
    │   │   ├── __init__.py       # LinuxGnomeAdapter
    │   │   ├── aliases.py        # APP_ALIASES (terminal, browser, etc.)
    │   │   ├── browser_focus.py  # Activities sweep
    │   │   ├── cursor_theme.py   # gsettings advice
    │   │   └── models/
    │   │       ├── pointer_accel/{config.json,weights.npz}
    │   │       └── longjump/{config.json,weights.npz}
    │   └── macos/                # MacOSAdapter (Cmd+Space + Ctrl→Cmd remap)
    ├── targets/                  # TargetRegistry, Target dataclass, TOML loader
    ├── ui/                       # FastAPI Command Center
    │   ├── server.py             # routes (1700+ lines, port from cc)
    │   ├── runner.py             # one-at-a-time ControllerAgent runner
    │   ├── factory.py            # make_target_context_factory
    │   ├── frame_store.py        # poll watch dir, SSE
    │   ├── log_bus.py            # SSE log capture
    │   ├── paste_protocol.py     # file paste with sha256 verify
    │   └── static/               # mobile-first SPA (index.html, app.js, styles.css)
    ├── pi/                       # Pi-side deployable (own venv on Pi)
    │   ├── server.py             # FastAPI HID gateway
    │   ├── bt_hid.py             # raw L2CAP + Profile1
    │   ├── hid_writer.py         # USB HID gadget fallback
    │   └── hid_codes.py
    └── cli.py                    # argparse: do, platforms, targets, vault, cc, version
```

## Recent (in-session) state

This session's work, in commit order on `origin/main`:

- `b69d7a4` — homer: retrain pointer_accel-yaru-v4. 232 click_at across grid + targeted regions → 2333 rows → hidden=64 model. Canary median 13 → 8.5 steps, max 17 → 13. Added best-val early-stop to trainer + new `collect_targeted.sh` script.
- `d921f57` — fix: Unlock — clear stale cached vault passphrase on failure. Always show modal; pollRunStatus wipes _vaultPassphraseCache on any failed unlock run.
- `200fe00` — fix: Unlock modal — "Skip visual verify" override. Threads `verify=False` into login PlanStep through RunRequest → runner → ControllerAgent → LoginAgent.
- `5a27a38` — fix: Unlock — replace window.prompt() with inline modal (browser-block-proof). Two tabs initially: via vault / type directly. Password fields type=password.
- `973f7b3` — feat: Unlock modal — "create new vault" tab. POST /api/vault/create wipes legacy + handsneyes vault files, creates fresh with new master passphrase + seed entry, then fires unlock with the new credentials.
- `c1b4b3c` — fix: Unlock end-to-end. Five linked fixes: UI prompts inline, RunRequest carries password + vault_passphrase, runner builds per-request Vault, controller threads password into login step, plan strings + RunRecord.public() redact secrets. Bonus: fixed hard-coded vision_model "nemotron-3-nano-omni" → "nvidia/nemotron-3-nano-omni".
- `6f14609` — feat: OCR sync from host text field (📥 button + auto-fire on click_at). nanonets-ocr-s on cropped ROI band.
- `b55ec69` — fix: passthrough field — Left/Right arrows work, typed text mirrors. Re-implemented local caret editing for Backspace/Delete/Left/Right/Home/End/Tab/Enter/Escape, removed focus gate so global hover capture also mirrors.
- `11d695a` — fix: arrow buttons + homer now finds shipped Ubuntu checkpoints. Added ↑↓←→ to keystroke row. Homer's `_POINTER_ACCEL_CHECKPOINT_CANDIDATES` now searches platform-bundled path first.
- `5df130d` — homer: retrain pointer_accel on default Yaru white cursor (yaru-v2). Cursor reset from `redglass` to default Yaru via gsettings; oscillation-variance + frame-diff detection validated.

## Pending / next

- Visual cue when global hover-capture is active (a thin border around the image — exists, but verify it actually paints).
- Make `--frames-dir` flag on cc useful (currently passes through but FrameStore default is `~/.local/share/handsneyes/runs/` regardless).
- Tests for the new `/api/vault/create` and `/api/sync-text-from-host` endpoints.
- `longjump-yaru-v3` retrain — current shipped longjump is the original redglass-trained one. Cursor-independent in principle (HID→pixel dynamics) but quality unverified on Yaru.
- Multi-target dropdown in cc UI.
- Webcam auto-detect: try every cv2 device on cc startup, pick the one returning non-SMPTE / non-black frames.
- macOS adapter: train pointer_accel + longjump checkpoints (currently no models ship — relies on closed-loop refinement only).
