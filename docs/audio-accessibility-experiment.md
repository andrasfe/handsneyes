# Audio + HID in parallel over one Bluetooth link

**Status:** validated experimentally, 2026-07. Not yet productised.

## The idea

handsneyes drives a target machine's mouse/keyboard over Bluetooth HID. The
same Bluetooth link can *simultaneously* carry audio in both directions. That
turns the setup into a general AI-assisted accessibility bridge:

```
                  one BT pairing, ZERO software on the target
   ┌──────────────┐  HID (mouse/kb)  ───────────────▶  ┌──────────────┐
   │  Linux host  │  A2DP  ◀──── target audio out ───  │   target     │
   │  + AI agent  │  HFP mic ───▶ target audio in ───▶ │  (locked-    │
   └──────────────┘                                    │   down OK)   │
                                                       └──────────────┘
```

- **Control** — the agent moves the cursor and types (already shipped).
- **Perception** — the target's audio arrives at the host, so screen-reader
  narration (VoiceOver, or a browser with accessibility enabled) becomes a
  *semantic* description of the screen. Run it through STT and the agent knows
  what's on screen without a webcam or OCR.
- **Speech** — the host presents as a microphone, so the agent can talk *into*
  the target: dictation fields, voice control, browser voice interfaces.

The target needs **no installed software and no admin rights** — it believes it
paired a mouse/keyboard/headset combo. That matters: users who need assistive
tech are often on managed machines where they cannot install anything.

## What was measured

With the local host acting as the HID device (see `local-hid-host.md`) and a
macOS target connected:

| Channel | PipeWire stream | Result |
|---|---|---|
| Control (host → target) | HID over L2CAP PSM 17/19 | 12/12 `move_large` delivered `ok` |
| Perception (target → host) | `bluez_input.<MAC>.0` | 6.10 s captured, RMS 739 / peak 7831 (real content, not silence) |
| Speech (host → target) | `bluez_output.<MAC>.1` ← host mic | routed and active (not exercised) |

Both ran **concurrently on the same ACL link** with no observed interference.
BR/EDR multiplexes the profiles; HID reports are tiny next to A2DP's
~300-400 kbps, well inside the EDR budget.

## Reproducing the capture

The audio arrives as a normal PipeWire stream. Capturing the *sink monitor*
requires an explicit property — without it you silently record zeros:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
wpctl status                      # find the sink id and the bluez_* streams
pw-record -P stream.capture.sink=true --target <SINK_ID> out.wav
```

Gotchas that cost time here:

- `pw-record --target <sink>` **without** `-P stream.capture.sink=true`
  produces a silent file and exits non-zero. Always verify a capture is
  non-silent (RMS) before drawing conclusions.
- Targeting the `bluez_input` *stream* node directly also yields silence — it
  is a playback stream feeding the sink, not a capture source.
- Run a **control test** (play a known tone, capture, check RMS) before
  believing a silent result means "no audio".
- `pactl` may not be installed even when PipeWire is running; use `wpctl` /
  `pw-dump` / `pw-record`.
- A non-interactive shell needs `XDG_RUNTIME_DIR` (and often
  `DBUS_SESSION_BUS_ADDRESS`) set, or the audio tools cannot reach PipeWire and
  appear to show "no audio devices".

## Note on the audio plugins

Audio flows **despite** `bluetoothd --noplugin=a2dp,media,audio`: PipeWire
registers its endpoints through BlueZ's D-Bus media API, independent of those
plugin flags. So that flag alone does not stop a target from routing its sound
to this host — the SDP strip (`scripts/bt-strip-audio-sdp.sh`) or a WirePlumber
rule is what actually suppresses it. For the accessibility use case the audio
is wanted, so the strip should be skipped or reversed.

## Local speech-to-text (validated)

The captured audio was transcribed **locally, on CPU**, closing the loop
target audio → Bluetooth → PipeWire capture → STT → text. `faster-whisper`
(CTranslate2, `int8`) on a Ryzen 7 5825U, 12 threads, no GPU:

| Model | Model load | Transcribe 6.1 s | Speed | Notes |
|---|---|---|---|---|
| `base.en` | 15.4 s | 2.24 s | 2.7× realtime | truncated the sentence |
| `small.en` | 45.7 s | 2.99 s | 2.0× realtime | complete, clean |

`small.en` is the recommended default: ~240 MB, comfortably faster than
realtime, and accurate on the sample. Load the model once and keep it resident
— the load cost is one-off, the per-utterance cost is what matters. Screen
reader narration is clean synthetic speech with no background noise, which is
an easier target than natural conversational audio.

```bash
pip install faster-whisper
```
```python
from faster_whisper import WhisperModel
model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=12)
segments, _ = model.transcribe("capture.wav", beam_size=5)
text = " ".join(s.text.strip() for s in segments)
```

## Open questions / next steps

1. **Isolate the perception stream.** The measurement above captured the sink
   monitor, i.e. the mix of everything playing locally. A real pipeline should
   capture the `bluez_input` node alone so local sound does not pollute STT.
2. **Stream instead of batch.** Feed STT rolling chunks (with VAD) rather than
   whole files, so narration turns into text as it is spoken.
3. **Prove the mic direction end-to-end.** macOS tends to switch its own output
   when HFP's microphone engages, which may fight the listening channel;
   expect to choreograph A2DP-for-listening vs HFP-only-while-speaking.
4. **Reconnect behaviour** is unchanged — macOS still resists an emulator's
   self-initiated reconnect after sleep (see `local-hid-host.md`).
