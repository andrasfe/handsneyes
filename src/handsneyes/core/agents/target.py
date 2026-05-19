"""TargetAgent — locate a click target on screen by description.

Cascade of locators (first hit wins):

  1. **OCR** (tesseract). When the user supplied a quoted token in the
     description, only that quoted text is matched — generic
     descriptors like "subreddit"/"entry" are kept out so they don't
     match the first generic occurrence on the page.
  2. **ShowUI on the full image** — prompted with several variants
     derived from the description (quoted substrings, Capitalised
     tokens, the literal description).
  3. **ShowUI on focused crops** (left sidebar, bottom sidebar, footer
     strip). Helps when the target is small text that ShowUI on the
     full image would miss.

Returns ``(x_pct, y_pct)`` in image-fraction coordinates.

NOTE: The terminaleyes implementation also had a "scene-map +
keyword-scored best match → ShowUI grounding" path layered between (1)
and (2), driven by the legacy ``ClosedLoopHomer``. That whole 990-LoC
homer is deferred to Phase B; the scene-map fallback rarely beats the
direct ShowUI prompts on real targets, so its absence is acceptable
for the Phase A smoke goal. The cropped + variant prompts still cover
the small-text case.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np  # noqa: TC002 — used by signatures inside the class body

from handsneyes.core.agents.base import Agent, Outcome
from handsneyes.core.vision.imaging import (
    enhance_for_screen,
    numpy_to_base64_png,
    resize_for_mllm,
)
from handsneyes.core.vision.ocr_finder import (
    annotate_ocr_hit,
    have_ocr,
)
from handsneyes.core.vision.ocr_finder import (
    find_text as ocr_find_text,
)
from handsneyes.core.vision.text_targeting import (
    showui_prompt_variants,
    target_keywords,
)

logger = logging.getLogger(__name__)


@dataclass
class TargetOutcome(Outcome):
    """``data['position'] = (x_pct, y_pct)`` and ``data['method']``."""


class TargetAgent(Agent):
    """Locate a target element by free-form description."""

    name = "target"

    async def run(
        self,
        *,
        description: str,
        image: np.ndarray | None = None,
        run_dir: Any = None,  # noqa: ANN401 — Path | None, typed loose
    ) -> TargetOutcome:
        if not description:
            return TargetOutcome(
                success=False, reason="empty description",
            )
        if image is None:
            if self.ctx.capture is None:
                return TargetOutcome(
                    success=False, reason="no capture in context",
                )
            frame = await self.ctx.capture.capture_frame()
            image = frame.image

        # 1. OCR — quoted-token primary search.
        if have_ocr():
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", description)
            primary = (
                [q.lower() for q in quoted]
                if quoted else target_keywords(description)
            )
            hits = ocr_find_text(image, primary)
            if hits:
                top = hits[0]
                if run_dir is not None:
                    try:
                        cv2.imwrite(
                            str(run_dir / "target_ocr_hit.png"),
                            annotate_ocr_hit(image, top),
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "Could not write target_ocr_hit.png",
                            exc_info=True,
                        )
                return TargetOutcome(
                    success=True,
                    reason=(
                        f"OCR matched {top.text!r} "
                        f"(conf={top.confidence:.0f})"
                    ),
                    data={
                        "position": (top.x_pct, top.y_pct),
                        "method": "ocr",
                        "matched_text": top.text,
                        "confidence": top.confidence,
                    },
                )

        b64 = await self._encode(image)

        # 2. ShowUI direct on the full image — variants from description.
        for p in showui_prompt_variants(description):
            pos = await self._showui_query(b64, p)
            if pos is not None:
                return TargetOutcome(
                    success=True,
                    reason=f"ShowUI direct grounded via {p!r}",
                    data={"position": pos, "method": "showui_direct"},
                )

        # 3. Cropped ShowUI for small / sidebar text.
        crop_regions = [
            ("sidebar_full",   0.0, 0.0,  0.30, 1.0),
            ("sidebar_bottom", 0.0, 0.55, 0.32, 1.0),
            ("footer_strip",   0.0, 0.75, 1.0,  1.0),
        ]
        ih, iw = image.shape[:2]
        quoted_for_crop = re.findall(r"['\"]([^'\"]+)['\"]", description)
        token = (
            quoted_for_crop[0]
            if quoted_for_crop else description.split()[-1]
        )
        for name, x0f, y0f, x1f, y1f in crop_regions:
            x0, y0 = int(x0f * iw), int(y0f * ih)
            x1, y1 = int(x1f * iw), int(y1f * ih)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_b64 = await self._encode(crop)
            for cp in (
                f"Click on {token}",
                f"Click on the {token} link",
                f"Click on r/{token}",
            ):
                pos = await self._showui_query(crop_b64, cp)
                if pos is not None:
                    fx = (pos[0] * (x1 - x0) + x0) / iw
                    fy = (pos[1] * (y1 - y0) + y0) / ih
                    return TargetOutcome(
                        success=True,
                        reason=(
                            f"ShowUI grounded {cp!r} on crop {name!r}"
                        ),
                        data={
                            "position": (fx, fy),
                            "method": "cropped_showui",
                            "crop": name,
                        },
                    )
            if have_ocr():
                hits_crop = ocr_find_text(
                    crop, [token.lower()],
                    crops=[(0.0, 0.0, 1.0, 1.0)],
                )
                if hits_crop:
                    top = hits_crop[0]
                    fx = (top.x_pct * (x1 - x0) + x0) / iw
                    fy = (top.y_pct * (y1 - y0) + y0) / ih
                    return TargetOutcome(
                        success=True,
                        reason=(
                            f"OCR on crop {name!r} matched "
                            f"{top.text!r}"
                        ),
                        data={
                            "position": (fx, fy),
                            "method": "cropped_ocr",
                            "crop": name,
                        },
                    )

        return TargetOutcome(
            success=False,
            reason="OCR + ShowUI direct + cropped ShowUI all missed",
        )

    # ───────────────────── helpers ─────────────────────

    async def _showui_query(self, b64: str, prompt: str) -> Any:  # noqa: ANN401
        if self.ctx.showui_query is None:
            return None
        try:
            return await self.ctx.showui_query(b64, prompt)
        except Exception as e:  # noqa: BLE001
            logger.debug("ShowUI query failed: %s", e)
            return None

    @staticmethod
    async def _encode(image: np.ndarray) -> str:
        resized = resize_for_mllm(
            enhance_for_screen(image),
            max_dimension=1280,
            min_dimension=768,
        )
        return numpy_to_base64_png(resized)
