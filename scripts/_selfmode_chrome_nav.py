"""Self-mode Chrome nav, robust to a cluttered desktop:
  1. open_app Chrome (front)
  2. homer-click an EMPTY area of the Chrome page -> Chrome becomes the
     key window (beats the focus war) WITHOUT focusing an on-page field
  3. Cmd+L -> omnibox, Cmd+A -> select, type URL, Enter
  4. capture for verification
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import cv2

from handsneyes.core.agents.context import AgentContext
from handsneyes.core.capture.screen import ScreenCapture
from handsneyes.core.vision.session_adapter import SessionAdapter
from handsneyes.core.vision.visual_servo_homer import VisualServoHomer
from handsneyes.io.keyboard import HttpKeyboardOutput, PlatformKeyboard
from handsneyes.io.mouse import HttpMouseOutput
from handsneyes.platforms import load_adapter
from handsneyes.platforms.base import AppHint
from handsneyes.platforms.macos.cursor_reader import QuartzCursorReader

PI = "http://10.0.0.2:8080"


async def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.linkedin.com/in/me/"
    # Default activation target = Chrome's DOCK icon (never occluded,
    # reliably makes Chrome the key window — beats the focus war).
    ax = float(sys.argv[2]) if len(sys.argv) > 2 else 0.639
    ay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.965
    wait_max = float(sys.argv[4]) if len(sys.argv) > 4 else 16.0

    adapter = load_adapter("macos")
    raw_kb = HttpKeyboardOutput(base_url=PI, transport="bt")
    kb = PlatformKeyboard(raw_kb, adapter)
    mouse = HttpMouseOutput(base_url=PI, transport="bt")
    await raw_kb.connect()
    await mouse.connect()
    cap = ScreenCapture(display_index=0)
    await cap.open()
    reader = QuartzCursorReader()

    out_dir = Path.home() / ".local/share/handsneyes/runs" / (
        "chromenav_" + datetime.now().strftime("%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / "page.png"

    ctx = AgentContext(
        mouse=mouse, keyboard=kb, capture=cap, cursor_reader=reader,
        vision_client=None, output_dir=out_dir, platform=adapter,
    )
    homer = VisualServoHomer(session=SessionAdapter(ctx))

    print(f"[cn] homer-click Chrome DOCK icon ({ax},{ay})", flush=True)
    out = await homer.home_to_pixel(ax, ay, click=True)
    print(f"[cn] click reason={out.reason} steps={out.steps}", flush=True)
    await asyncio.sleep(0.8)

    # Chrome is now the key window. Focus the OMNIBOX directly (Cmd+L
    # selects all existing text) and type — NO new tab. The Cmd+T new
    # tab dropped focus into the New Tab Page's central Google box,
    # which searches everything; the omnibox treats https:// as a URL.
    await kb.send_key_combo(["ctrl"], "l")   # -> meta+l (focus omnibox)
    await asyncio.sleep(0.5)
    print(f"[cn] type {url}", flush=True)
    # warmup=False: the Pi's first-char double-tap+backspace warmup
    # leaves a DOUBLED first character in a URL bar (Backspace there
    # doesn't delete a char), producing "hhttps://…" which Chrome
    # treats as a search query instead of a URL.
    await kb.send_text(url, warmup=False)
    await asyncio.sleep(0.8)
    await kb.send_keystroke("Enter")

    print("[cn] loading…", flush=True)
    prev = None
    for i in range(int(wait_max / 0.5)):
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
    for c in (raw_kb, mouse):
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
