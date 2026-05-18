# handsneyes

Vision-based agentic terminal controller. Successor to [terminaleyes](https://github.com/andrasfe/terminaleyes).

Webcam captures a target machine's screen; classical CV + multimodal LLMs locate the cursor and the click target; HID commands flow over BT (or USB) via a Raspberry Pi to drive the target machine.

**v2.0 goals over terminaleyes:**

- Per-OS support is a plugin module (`platforms/linux_gnome`, `platforms/macos`, …). Adding a new OS = adapter class + lightweight per-OS model + one entry-point line.
- One Command Center UI manages multiple named target hosts (one active at a time).
- Lightweight ML models (pointer-acceleration, long-jump) ship inside their platform module, with a documented retraining workflow for new OSes.

Architecture plan: see [`docs/architecture.md`](docs/architecture.md). Porting to a new OS: see [`docs/porting-to-new-os.md`](docs/porting-to-new-os.md).

Status: bootstrapping. The actual port from terminaleyes happens in three phases (see architecture doc).
