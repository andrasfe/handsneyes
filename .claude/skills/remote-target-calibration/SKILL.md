---
name: remote-target-calibration
description: Calibrate, verify, and troubleshoot handsneyes mouse precision against a remote target (especially macOS over HDMI-capture + BT HID). Use when clicks land off-target, when bringing up a new target machine, when the cursor/display/accel configuration changed, or when asked to measure click precision.
---

# Remote-target calibration & precision verification

handsneyes' click precision depends on several host-specific
calibrations. They are all **self-acquired** (no operator
measurement) and **persisted per target** — your job is usually just
to verify them, invalidate the stale ones, and re-measure.

## The calibration inventory

| what | acquired | persisted at | staleness |
|---|---|---|---|
| openloop pct-per-HID ratio (x, y) | first click_at of a session: slam to corner + 1000-HID burst | `~/.config/handsneyes/homer_calibration_<target>.json` (payload[6:8]) | 30-day TTL; per-click corrections absorb drift |
| closed-loop + cruise ratios, hotspot offset | EMA-learned during clicks | same file (payload[0:6]) | same |
| pointer-accel scale knob | operator-set in cc UI | `~/.config/handsneyes/pointer_accel_scale.json` | manual; **reset to 1.0/1.0 when the target changes** — a scale tuned for one display skews every bulk burst on another |
| screen homography (webcam keystone) | auto: screen-quad detection at capture open | `~/.config/handsneyes/rectify_<target>.json` | re-detected each session; cache is dark-screen fallback. HDMI capture needs none (passes through) |
| pointer-accel / longjump ML models | offline training | `src/handsneyes/platforms/<os>/models/` | retrain rules in CLAUDE.md |

Delete the relevant JSON file to force a fresh calibration.

## How a click converges (what the logs mean)

With `HANDSNEYES_OPENLOOP=1` every click runs: calibrate-ratio (once)
→ slam or cached-position start → bulk burst → up to 8 correction
iterations → click. Each iteration localizes the cursor by, in
priority order:

1. **stock-template match** (macOS only; pixel-precise, returns the
   hotspot directly) — log line `template hit (macos-arrow-13, …)`,
   corrections marked `[trusted]` may burst up to 2500 HID;
2. **frame-diff** vs the previous iteration's frame, elected by
   predicted position — untrusted, clamped to ±400 HID;
3. **oscillation-variance** (slow jiggle) as last resort.

Healthy convergence reaches `within tol` (0.4 % ≈ 8 px) in 2-5
iterations. Watch for:

- `diff phantom, disabling diff localize` — the diff latched onto a
  static UI change (focus highlight, caret). Normal & self-healing.
- `cursor not moving despite bursts — committing` — HID isn't
  reaching the host. Check `bt_hid_connected` on the Pi.
- `hit 8 iters without convergence (residual=X%)` — the click fired
  off-target. If frequent: ratio badly stale (delete the calibration
  file) or accel scale wrong (reset to 1.0).
- Template never hits on a target where the cursor IS the arrow —
  capture geometry off; check rectification / capture device.

## Verifying precision (the battery)

```bash
.venv/bin/python scripts/precision_battery.py \
    --points "0.25,0.60 0.65,0.30 0.70,0.70 0.25,0.25 0.62,0.55 0.28,0.80"
```

- Pick aim points on INERT surfaces from a fresh `/api/snapshot`
  frame: editor/terminal panes, desktop background. NEVER over
  buttons, dock icons, menu items, or settings toggles — every
  point is a real click on the target.
- The battery verifies landings **independently** via stock-cursor
  template matching on post-click frames. Do not trust the homer's
  own `residual` as the only evidence — it once reported success at
  a 99 px miss.
- `UNVERIFIED` rows = cursor at the landing is not the arrow
  (I-beam over text). Cross-check those against the cc log's final
  `within tol` line.
- Good numbers (1080p HDMI capture): median ≤ 5 px, max ≤ 10 px,
  8-12 s per click (first click +10 s for calibration).

## Bring-up order for a new target

1. `scripts/reconnect.sh` — USB ECM + Pi gateway + BT pairing flow.
   The Pi-side BT gotchas (stale bonds, macOS SDP caching, radio
   mode) are documented in CLAUDE.md and handled by the script.
2. `[[target]]` entry in `config/targets.toml` (platform, camera,
   `screen_size`; `rectify = true` only for real webcams).
3. Reset accel scale: `curl -X POST localhost:8765/api/pointer-accel-scale -H 'Content-Type: application/json' -d '{"scale_x":1.0,"scale_y":1.0}'`
4. Start cc: `HANDSNEYES_OPENLOOP=1 handsneyes cc --target <name>`
5. One throwaway click_at at a safe point (pays the calibration),
   then run the battery and read the convergence logs.
