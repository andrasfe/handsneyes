"""Self-mode: precision-click the Chrome address bar with the improved
oracle homer, then type a URL and capture. Sidesteps the keyboard-focus
problem by putting focus exactly where we click.
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
OMNI_X, OMNI_Y = 0.30, 0.295


async def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.linkedin.com/in/me/"
    wait_max = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

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
        "clicknav_" + datetime.now().strftime("%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / "page.png"

    ctx = AgentContext(
        mouse=mouse, keyboard=kb, capture=cap, cursor_reader=reader,
        vision_client=None, output_dir=out_dir, platform=adapter,
    )
    homer = VisualServoHomer(session=SessionAdapter(ctx))

    print("[clicknav] Chrome front", flush=True)
    await adapter.open_app(kb, app=AppHint(canonical="Google Chrome"), settle_ms=1500)

    print(f"[clicknav] homer-click omnibox ({OMNI_X},{OMNI_Y})", flush=True)
    out = await homer.home_to_pixel(OMNI_X, OMNI_Y, click=True)
    pos = await reader.read_pct()
    print(f"[clicknav] landed {pos} reason={out.reason} steps={out.steps}", flush=True)
    await asyncio.sleep(0.4)

    # Select-all in the (now focused) omnibox, then type + go.
    await kb.send_key_combo(["ctrl"], "a")  # → meta+a
    await asyncio.sleep(0.2)
    print(f"[clicknav] type {url}", flush=True)
    await kb.send_text(url)
    await asyncio.sleep(0.25)
    await kb.send_keystroke("Enter")

    print("[clicknav] loading…", flush=True)
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
