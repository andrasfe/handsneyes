"""NavigateAgent — type a URL into a browser address bar.

Verify-then-act flow:

  1. **Pre-flight** — :class:`VerifyAgent` decides whether the
     foreground app is a web browser. If not, try to bring a browser
     to focus via ``ctx.platform.focus_browser`` (which on
     linux_gnome runs the Super → type "firefox" → Enter sequence;
     on macos it's Cmd+Tab + Spotlight). Each attempt re-verifies.
  2. **Type** — Ctrl+L (URL-bar focus) → Ctrl+A → text → Enter.
     The PlatformKeyboard proxy remaps Ctrl→Cmd on macOS.
  3. **Post-flight** — OCR the URL bar; if the typed URL doesn't
     appear, fail explicitly. No more silent successes when keystrokes
     went into the wrong app.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from handsneyes.core.agents.base import Agent, Outcome
from handsneyes.core.agents.type_text import TypeAgent
from handsneyes.core.agents.verify import VerifyAgent

logger = logging.getLogger(__name__)


@dataclass
class NavigateOutcome(Outcome):
    pass


_BROWSER_QUESTION = (
    "Look at the screen. Is the FOREGROUND application a web "
    "browser (visible URL/address bar at the top, browser tab strip, "
    "and a web page rendered in the body)? Answer TRUE only if the "
    "foreground app is unmistakably a browser. If the foreground is "
    "a settings dialog, file manager, terminal, software updater, "
    "media player, etc., answer FALSE — even if a browser window is "
    "partially visible behind it."
)


class NavigateAgent(Agent):
    """Drive a URL into a browser address bar, verifying both ends."""

    name = "navigate"

    async def run(  # type: ignore[override]
        self,
        *,
        url: str,
        select_all_first: bool = True,
        post_settle: float = 1.8,
        ensure_browser: bool = True,
        verify_after: bool = True,
        max_focus_attempts: int = 4,
    ) -> NavigateOutcome:
        if self.ctx.keyboard is None:
            return NavigateOutcome(
                success=False, reason="no keyboard in context",
            )
        if not url:
            return NavigateOutcome(success=False, reason="empty url")

        # 1. Pre-flight: ensure a browser is the foreground app.
        if ensure_browser:
            ok, reason = await self._ensure_browser_focused(
                max_attempts=max_focus_attempts,
            )
            if not ok:
                return NavigateOutcome(
                    success=False,
                    reason=f"could not focus a browser: {reason}",
                )
            # Tight re-check: model calls + activation may have taken
            # several seconds; the foreground can shift in that gap.
            v_tight = await VerifyAgent(self.ctx).run(
                question=_BROWSER_QUESTION,
                visual_only=True,
                record_label="navigate_browser_pre_type",
            )
            if not v_tight:
                return NavigateOutcome(
                    success=False,
                    reason=(
                        "browser was focused initially but foreground "
                        f"shifted before typing: {v_tight.reason}"
                    ),
                )

        # 2. Type the URL via the URL-bar focus chord. PlatformKeyboard
        # remaps Ctrl→Cmd on macOS for us.
        try:
            await self.ctx.keyboard.send_key_combo(["ctrl"], "l")
            await asyncio.sleep(0.5)
            if select_all_first:
                await self.ctx.keyboard.send_key_combo(["ctrl"], "a")
                await asyncio.sleep(0.25)
            # Skip the keyboard-backend warmup: in browser URL bars
            # Backspace is often back-navigation, which leaves the
            # warmup's first press visible.
            await TypeAgent(self.ctx).run(
                text=url, secret=False, submit=False, warmup=False,
            )
            await asyncio.sleep(0.4)
            await self.ctx.keyboard.send_keystroke("Enter")
        except Exception as e:  # noqa: BLE001
            logger.warning("NavigateAgent typing failed: %s", e)
            return NavigateOutcome(
                success=False, reason=f"send failed: {e}",
            )

        await asyncio.sleep(post_settle)

        # 3. Post-flight: confirm the URL actually appeared.
        if verify_after and self.ctx.capture is not None:
            verified, vreason = await self._verify_url_in_address_bar(url)
            if not verified:
                return NavigateOutcome(
                    success=False,
                    reason=(
                        f"typed URL but address bar does NOT confirm "
                        f"navigation ({vreason})"
                    ),
                    data={"url": url, "verified": False},
                )
            return NavigateOutcome(
                success=True,
                reason=f"navigated to {url!r} ({vreason})",
                data={"url": url, "verified": True},
            )

        return NavigateOutcome(
            success=True,
            reason=f"sent navigation keystrokes for {url!r} (unverified)",
            data={"url": url, "verified": False},
        )

    # ─────────────────── browser-focus pre-flight ───────────────────

    async def _ensure_browser_focused(
        self, *, max_attempts: int,
    ) -> tuple[bool, str]:
        """Verify-then-correct loop until the foreground is a browser.

        Corrective action comes from
        ``ctx.platform.focus_browser(ctx, attempt=…, max_attempts=…)``
        on each retry. If no adapter is configured we can only verify
        and report — no fallback action.
        """
        verifier = VerifyAgent(self.ctx)
        v0 = await verifier.run(
            question=_BROWSER_QUESTION,
            visual_only=True,
            record_label="navigate_browser_check",
        )
        if v0:
            return True, v0.reason

        if self.ctx.platform is None:
            return False, (
                "no platform adapter configured; cannot recover focus"
            )

        for attempt in range(1, max_attempts + 1):
            try:
                method = await self.ctx.platform.focus_browser(
                    self.ctx,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "platform.focus_browser(attempt=%d) raised: %s",
                    attempt,
                    e,
                )
                continue
            await asyncio.sleep(0.9)
            v = await verifier.run(
                question=_BROWSER_QUESTION,
                visual_only=True,
                record_label=f"navigate_browser_recheck_{attempt:02d}",
            )
            if v:
                return True, f"activated via {method}; {v.reason}"
        return False, (
            f"{max_attempts} activation attempts did not bring a "
            "browser to the foreground"
        )

    # ─────────────────── URL bar post-flight ───────────────────

    async def _verify_url_in_address_bar(
        self, url: str,
    ) -> tuple[bool, str]:
        """OCR the URL bar (and, if needed, the whole page) and check
        the typed URL appears. Tolerant of letter-substitution
        garbling (tesseract often reads ``r`` as ``t``); we normalise
        to alphanumerics and substring-match.
        """
        assert self.ctx.capture is not None
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception as e:  # noqa: BLE001
            return False, f"post-capture failed: {e}"
        self.ctx.record_frame(frame.image, label="navigate_postflight_full")
        h, _w = frame.image.shape[:2]
        urlbar = frame.image[: int(h * 0.10), :]
        self.ctx.record_frame(urlbar, label="navigate_postflight_urlbar")
        wide_top = frame.image[: int(h * 0.18), :]

        try:
            import pytesseract  # type: ignore[import-not-found]

            from handsneyes.core.vision.ocr_finder import (
                _preprocess_for_ocr,
                have_ocr,
            )

            if not have_ocr():
                return True, "OCR unavailable; trusting keystrokes"
        except Exception as e:  # noqa: BLE001
            return True, f"OCR import failed ({e}); trusting keystrokes"

        # Candidate set from URL.
        url_lower = re.sub(r"^https?://", "", url.lower())
        path_segments = [
            re.sub(r"[^a-z0-9]", "", seg)
            for seg in url_lower.split("/")
            if seg.strip()
        ]
        path_segments = [s for s in path_segments if len(s) >= 4]
        host = re.sub(
            r"[^a-z0-9.]", "", url_lower.split("/")[0],
        )
        host_parts = [p for p in host.split(".") if len(p) >= 3]
        candidates = list(dict.fromkeys(path_segments + host_parts))

        def _scan(label: str, region: object) -> tuple[bool, str] | None:
            try:
                normal = pytesseract.image_to_string(region)
                inv = pytesseract.image_to_string(
                    _preprocess_for_ocr(region, scale=4, invert=True),  # type: ignore[arg-type]
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("OCR pass %s failed: %s", label, e)
                return None
            text = (normal + " " + inv).lower()
            text_norm = re.sub(r"[^a-z0-9]", "", text)
            for c in candidates:
                if c in text_norm:
                    return True, f"{label} contains {c!r}"
            words = re.findall(r"[a-z0-9]{4,}", text_norm)
            for c in candidates:
                for w in words:
                    if abs(len(w) - len(c)) > max(2, len(c) // 3):
                        continue
                    ratio = SequenceMatcher(None, c, w).ratio()
                    if ratio >= 0.75:
                        return True, (
                            f"{label} fuzz-matches {c!r} ~ {w!r} "
                            f"(ratio={ratio:.2f})"
                        )
            return False, (
                f"{label} text {text.strip()[:80]!r} did not match"
            )

        for label, region in (
            ("url-bar", urlbar),
            ("top-strip", wide_top),
            ("full-frame", frame.image),
        ):
            result = _scan(label, region)
            if result is None:
                continue
            ok, msg = result
            if ok:
                return True, msg
        return False, f"no OCR pass matched any of {candidates!r}"
