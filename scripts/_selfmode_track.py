"""Self-mode task: drive THIS mac via handsneyes (BT HID + macOS
adapter) to open the FedEx tracking page for a number, then
self-capture the result for reading.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import cv2

from handsneyes.core.capture.screen import ScreenCapture
from handsneyes.io.keyboard import HttpKeyboardOutput, PlatformKeyboard
from handsneyes.platforms import load_adapter
from handsneyes.platforms.base import AppHint

PI = "http://10.0.0.2:8080"
TRK = "818722860862"
URL = f"https://www.fedex.com/fedextrack/?trknbr={TRK}"


async def main() -> int:
    adapter = load_adapter("macos")
    raw_kb = HttpKeyboardOutput(base_url=PI, transport="bt")
    kb = PlatformKeyboard(raw_kb, adapter)
    await raw_kb.connect()
    cap = ScreenCapture(display_index=0)
    await cap.open()

    out_dir = Path.home() / ".local/share/handsneyes/runs" / (
        "fedex_" + datetime.now().strftime("%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[fetch] Safari via Spotlight", flush=True)
    await adapter.open_app(kb, app=AppHint(canonical="Safari"), settle_ms=1400)
    # Minimal legs: focus address bar in the current window, type, go.
    # ctrl+l is the cross-platform idiom; the macOS adapter remaps it
    # to meta+l (Cmd+L) at the IO boundary.
    await kb.send_key_combo(["ctrl"], "l")
    await asyncio.sleep(0.3)
    print(f"[fetch] typing {URL}", flush=True)
    await kb.send_text(URL)
    await asyncio.sleep(0.2)
    await kb.send_keystroke("Enter")

    # Poll-capture until the frame stops changing (page settled) or cap.
    print("[fetch] loading…", flush=True)
    prev = None
    shot = out_dir / "fedex.png"
    for i in range(16):  # ~8s max
        await asyncio.sleep(0.5)
        img = cap._grab_sync()
        cv2.imwrite(str(shot), img)
        if prev is not None:
            d = float((cv2.absdiff(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
            ) > 18).mean())
            if i >= 6 and d < 0.002:
                break
        prev = img
    print(str(shot), flush=True)
    try:
        await raw_kb.disconnect()
    except Exception:
        pass
    try:
        await cap.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
