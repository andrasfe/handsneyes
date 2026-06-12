# AGENTS.md — for AI agents working in this repo

Orientation for any coding agent (Claude Code, Codex, Cursor, ...).
`CLAUDE.md` is the deep reference (architecture, layering rules, ML
pipeline, BT HID gotchas); this file is the index.

## Skills (procedural knowledge — read these before reinventing)

Skills live under `.claude/skills/`. They are plain markdown: if your
harness doesn't auto-load them, just read the file.

| skill | when to use |
|---|---|
| [`remote-target-calibration`](.claude/skills/remote-target-calibration/SKILL.md) | Clicks land off-target; bringing up a new target machine; display/cursor/accel config changed; asked to measure or verify mouse precision. Covers the calibration inventory (what's persisted where, what to delete to force recalibration), how to read homer convergence logs, and the independent precision battery (`scripts/precision_battery.py`). |

## handsneyes agent classes (`src/handsneyes/core/agents/`)

Tiered; each returns `Outcome {success, reason, data}` and shares one
`AgentContext` (capture, mouse, keyboard, vision client, vault,
platform adapter, output dir) per session.

| tier | agent | role |
|---|---|---|
| 1 (atomic) | `VerifyAgent` | visual yes/no via multimodal LLM |
| 1 | `CursorAgent` | locate cursor (HSV → oscillation-variance cascade) |
| 1 | `TargetAgent` | locate click target (OCR → ShowUI grounding) |
| 2 (actions) | `WakeAgent` | wake display (jiggle/arrow/click + brightness check) |
| 2 | `TypeAgent` | text input, `secret=True` redaction |
| 2 | `ScrollAgent` | mouse-wheel scroll, optional hover-at |
| 3 (workflows) | `FocusAgent` | centre app via adapter `window_action` |
| 3 | `LoginAgent` | wake + verify + type, vault or direct password |
| 3 | `NavigateAgent` | browser-aware URL typing with post-OCR oracle |
| 3 | `ClickAgent` | wraps `VisualServoHomer` with scroll-and-retry |
| 4 (storage) | `Vault` | AES-256-GCM credential file |
| — | `ControllerAgent` | decomposes free-form intents into PlanSteps |

## Ground rules (the load-bearing ones)

- Layering: `core/` never imports `platforms/<anything>` or `pi/`;
  platform behaviour reaches core only through the injected
  `PlatformAdapter`. Adding an OS must be strictly additive. Details
  in CLAUDE.md.
- `pi/` is its own deployable (runs on the Raspberry Pi gateway);
  deploy with `scripts/deploy_pi.sh` — the Pi usually has NO internet
  (offline installer handles it).
- Never trust the homer's own residual as proof a click landed —
  verify independently (template match / precision battery).
- Tests: `python -m pytest tests/ -v` must stay green.

## Quick commands

```bash
pip install -e ".[dev]"                                  # setup
python -m pytest tests/ -q                               # tests
HANDSNEYES_OPENLOOP=1 handsneyes cc --target <name>      # command center
scripts/reconnect.sh                                     # Pi/BT bring-up
.venv/bin/python scripts/precision_battery.py            # precision check
```
