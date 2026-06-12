"""Self-mode nav+capture: open <app> via Spotlight, go to <url>,
poll-capture until the page settles, print the screenshot path.

Usage: _selfmode_nav.py "<app canonical>" "<url>" [wait_max_s]
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


async def main() -> int:
    app = sys.argv[1] if len(sys.argv) > 1 else "Google Chrome"
    url = sys.argv[2] if len(sys.argv) > 2 else "https://www.linkedin.com"
    wait_max = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0

    adapter = load_adapter("macos")
    raw_kb = HttpKeyboardOutput(base_url=PI, transport="bt")
    kb = PlatformKeyboard(raw_kb, adapter)
    await raw_kb.connect()
    cap = ScreenCapture(display_index=0)
    await cap.open()

    out_dir = Path.home() / ".local/share/handsneyes/runs" / (
        "nav_" + datetime.now().strftime("%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / "page.png"

    print(f"[nav] {app} via Spotlight", flush=True)
    await adapter.open_app(kb, app=AppHint(canonical=app), settle_ms=2000)
    # Cmd+N opens a brand-new window so Chrome is the key window
    # (cluttered multi-window setup otherwise leaves focus in an on-page
    # field). The new window shows the New Tab Page whose CENTRAL search
    # box routes to Google — so explicitly grab the OMNIBOX with Cmd+L
    # (which DOES treat https:// as a URL) before typing.
    await kb.send_key_combo(["ctrl"], "n")
    await asyncio.sleep(1.1)
    await kb.send_key_combo(["ctrl"], "l")
    await asyncio.sleep(0.5)
    print(f"[nav] go: {url}", flush=True)
    await kb.send_text(url)
    await asyncio.sleep(0.25)
    await kb.send_keystroke("Enter")

    print("[nav] loading…", flush=True)
    prev = None
    steps = int(wait_max / 0.5)
    for i in range(steps):
        await asyncio.sleep(0.5)
        img = cap._grab_sync()
        cv2.imwrite(str(shot), img)
        if prev is not None:
            d = float((cv2.absdiff(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
            ) > 18).mean())
            if i >= 6 and d < 0.0015:
                break
        prev = img
    print(str(shot), flush=True)
    for c in (raw_kb,):
        try:
            await c.disconnect()
        except Exception:
            pass
    try:
        await cap.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
