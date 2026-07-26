#!/usr/bin/env python3
"""speech_listener.py — transcribe the target machine's audio, locally.

The target's sound arrives over the same Bluetooth link that carries HID
control (see docs/audio-accessibility-experiment.md). This service taps that
audio, waits until someone/something is actually speaking, transcribes each
utterance with a local Whisper model, and appends the text to speech_logs/.

Nothing is sent anywhere: capture, VAD and STT all run on this machine.

Why a port tap rather than a plain recording: the target's audio appears in
PipeWire as `bluez_input.<MAC>.N`, which is a *playback stream* (it feeds the
speakers), not a capture source — recording it directly yields silence. So we
start a capture client with autoconnect disabled and link the stream's output
ports to it. That is additive: existing links are untouched, so the audio keeps
playing normally while we listen.

Usage:
    python3 scripts/speech_listener.py                 # run until Ctrl-C
    python3 scripts/speech_listener.py --model base.en --threshold 250
    python3 scripts/speech_listener.py --list          # show candidate sources

Output (default <repo>/speech_logs/, override with --log-dir or
$HANDSNEYES_SPEECH_LOG_DIR):
    speech_logs/YYYY-MM-DD.jsonl   one JSON object per utterance
    speech_logs/YYYY-MM-DD.txt     human-readable "[HH:MM:SS] text"
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RATE = 16000          # Whisper's native rate — no resampling later
FRAME_MS = 30
FRAME_BYTES = int(RATE * FRAME_MS / 1000) * 2      # s16 mono
TAP_NAME = "handsneyes-speech-tap"

# Whisper emits these for silence/noise when nothing was really said.
_JUNK = {
    "", "you", "thank you", "thanks for watching", "bye", ".", "!", "?",
    "thank you.", "you.", "bye.", "thanks.",
}


def _env() -> dict:
    """PipeWire needs the user session's runtime dir + bus; a non-interactive
    shell often lacks them and the tools then report no devices at all."""
    e = dict(os.environ)
    uid = os.getuid()
    e.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    e.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
    return e


def find_source_ports(pattern: str) -> list[str]:
    """Return output ports whose node matches `pattern` (e.g. 'bluez_input')."""
    try:
        out = subprocess.check_output(
            ["pw-link", "-o"], env=_env(), timeout=5,
        ).decode("utf-8", "replace")
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    ports = [ln.strip() for ln in out.splitlines() if re.search(pattern, ln)]
    # One channel is enough: we transcribe mono, and both carry the same speech.
    fl = [p for p in ports if p.endswith("_FL")]
    return fl or ports[:1]


class Listener:
    def __init__(self, args) -> None:
        self.args = args
        self.log_dir = Path(args.log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.model = None

    # -- logging ---------------------------------------------------------
    def _write(self, text: str, duration: float, peak: float) -> None:
        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        rec = {
            "ts": now.isoformat(timespec="seconds"),
            "duration_s": round(duration, 2),
            "peak_rms": round(peak, 1),
            "text": text,
        }
        with (self.log_dir / f"{day}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with (self.log_dir / f"{day}.txt").open("a", encoding="utf-8") as f:
            f.write(f"[{now.strftime('%H:%M:%S')}] {text}\n")
        print(f"[{now.strftime('%H:%M:%S')}] {text}", flush=True)

    # -- transcription worker --------------------------------------------
    def _worker(self) -> None:
        from faster_whisper import WhisperModel

        print(f"loading model {self.args.model} (one-off)...", flush=True)
        self.model = WhisperModel(
            self.args.model, device="cpu", compute_type="int8",
            cpu_threads=self.args.threads,
        )
        print("model ready — listening (only transcribes when sound arrives)",
              flush=True)

        while not self.stop.is_set():
            try:
                audio, duration, peak = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                segments, _ = self.model.transcribe(
                    audio, beam_size=5, language="en",
                    # Whisper invents text on silence; VAD already gated us,
                    # this is the second line of defence.
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:  # noqa: BLE001
                print(f"  transcribe failed: {e}", file=sys.stderr, flush=True)
                continue
            if text.strip().lower().rstrip(".!?") in _JUNK or not text:
                continue
            self._write(text, duration, peak)

    # -- capture + VAD ----------------------------------------------------
    def run(self) -> int:
        ports = find_source_ports(self.args.source)
        if not ports:
            print(f"no audio source matching {self.args.source!r}. Is the "
                  f"target connected and playing? Try --list.", file=sys.stderr)
            return 2
        print(f"tapping: {ports[0]}", flush=True)

        rec = subprocess.Popen(
            ["pw-record", "-P", "node.autoconnect=false",
             "-P", f"node.name={TAP_NAME}",
             "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_env(),
        )
        # Give the client time to appear in the graph, then link the tap.
        linked = False
        for _ in range(20):
            time.sleep(0.25)
            try:
                ins = subprocess.check_output(
                    ["pw-link", "-i"], env=_env(), timeout=5,
                ).decode("utf-8", "replace")
            except subprocess.SubprocessError:
                continue
            tgt = [ln.strip() for ln in ins.splitlines() if TAP_NAME in ln]
            if tgt:
                subprocess.run(["pw-link", ports[0], tgt[0]],
                               env=_env(), capture_output=True, timeout=5)
                linked = True
                break
        if not linked:
            print("could not link the tap (is pipewire running?)",
                  file=sys.stderr)
            rec.terminate()
            return 3

        threading.Thread(target=self._worker, daemon=True).start()

        # VAD state machine: only buffer while sound is arriving; flush the
        # utterance once it goes quiet again.
        buf: list[bytes] = []
        speaking = False
        silent_frames = 0
        voiced_frames = 0
        peak = 0.0
        need_silence = int(self.args.hang_ms / FRAME_MS)
        min_frames = int(self.args.min_ms / FRAME_MS)
        max_frames = int(self.args.max_s * 1000 / FRAME_MS)

        def flush() -> None:
            nonlocal buf, speaking, silent_frames, voiced_frames, peak
            if speaking and len(buf) >= min_frames:
                raw = b"".join(buf)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                audio /= 32768.0
                self.q.put((audio, len(buf) * FRAME_MS / 1000.0, peak))
            buf, speaking, silent_frames, voiced_frames, peak = [], False, 0, 0, 0.0

        try:
            while not self.stop.is_set():
                chunk = rec.stdout.read(FRAME_BYTES)
                if not chunk or len(chunk) < FRAME_BYTES:
                    if rec.poll() is not None:
                        print("capture ended (target disconnected?)",
                              file=sys.stderr)
                        break
                    continue
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples * samples)))
                loud = rms >= self.args.threshold

                if loud:
                    voiced_frames += 1
                    peak = max(peak, rms)
                    silent_frames = 0
                    if not speaking and voiced_frames >= 2:
                        speaking = True          # sound is arriving — activate
                    if speaking:
                        buf.append(chunk)
                else:
                    voiced_frames = 0
                    if speaking:
                        buf.append(chunk)        # keep trailing silence for context
                        silent_frames += 1
                        if silent_frames >= need_silence:
                            flush()
                if speaking and len(buf) >= max_frames:
                    flush()
        except KeyboardInterrupt:
            pass
        finally:
            flush()
            self.stop.set()
            rec.terminate()
            try:
                rec.wait(timeout=3)
            except subprocess.TimeoutExpired:
                rec.kill()
            time.sleep(1.0)   # let an in-flight transcription finish writing
        return 0


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="bluez_input",
                    help="regex matching the PipeWire output port to tap "
                         "(default: bluez_input — the Bluetooth target)")
    ap.add_argument("--model", default="small.en",
                    help="faster-whisper model (default small.en; base.en is "
                         "faster, medium.en more accurate)")
    ap.add_argument("--log-dir",
                    default=os.environ.get("HANDSNEYES_SPEECH_LOG_DIR",
                                           str(repo / "speech_logs")))
    ap.add_argument("--threshold", type=float, default=200.0,
                    help="RMS above which audio counts as sound (s16 scale)")
    ap.add_argument("--hang-ms", type=int, default=800,
                    help="silence that ends an utterance")
    ap.add_argument("--min-ms", type=int, default=400,
                    help="ignore blips shorter than this")
    ap.add_argument("--max-s", type=int, default=30,
                    help="force-flush a very long utterance")
    ap.add_argument("--threads", type=int, default=max(2, os.cpu_count() // 2))
    ap.add_argument("--list", action="store_true",
                    help="list candidate audio sources and exit")
    args = ap.parse_args()

    if args.list:
        try:
            out = subprocess.check_output(["pw-link", "-o"], env=_env(),
                                          timeout=5).decode("utf-8", "replace")
            print(out)
        except Exception as e:  # noqa: BLE001
            print(f"could not list ports: {e}", file=sys.stderr)
            return 2
        return 0

    return Listener(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
