"""OCR-based target locator.

When the multimodal grounding model (ShowUI / gemma) misses a small
on-screen text target, fall back to classical OCR via Tesseract.
Tesseract reads each word and gives its bounding box; we then match
the user's target keywords against the OCR words and return the centre
of the best match in image-fraction coordinates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    import numpy as np

try:
    import pytesseract  # type: ignore
    _HAVE_TESSERACT = True
except ImportError:
    _HAVE_TESSERACT = False

logger = logging.getLogger(__name__)


@dataclass
class OCRHit:
    x_pct: float
    y_pct: float
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in image px


def have_ocr() -> bool:
    return _HAVE_TESSERACT


def _preprocess_for_ocr(
    image_bgr: np.ndarray, scale: int = 3, invert: bool = False,
) -> np.ndarray:
    """Upscale + grey + threshold. ``invert=True`` for white-text-on-dark."""
    h, w = image_bgr.shape[:2]
    scaled = cv2.resize(
        image_bgr, (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    grey = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    if invert:
        # Dark mode: white text on dark background. Invert so the
        # post-threshold image is dark-text-on-light, which tesseract
        # is trained for.
        grey = 255 - grey
    bw = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        31, 12,
    )
    return bw


def _ocr_words(
    image_bgr: np.ndarray, scale: int, psm: int, invert: bool = False,
) -> list[dict[str, Any]]:
    """Run tesseract once at given preprocessing and return word records."""
    pre = _preprocess_for_ocr(image_bgr, scale=scale, invert=invert)
    h_pre, w_pre = pre.shape[:2]
    try:
        data = pytesseract.image_to_data(
            pre, output_type=pytesseract.Output.DICT,
            config=f"--psm {psm}",
        )
    except Exception as e:
        logger.warning("Tesseract failed (psm=%d): %s", psm, e)
        return []
    n = len(data.get("text", []))
    out: list[dict[str, Any]] = []
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        out.append({
            "word": word, "conf": conf,
            "x": data["left"][i] / w_pre,
            "y": data["top"][i] / h_pre,
            "w": data["width"][i] / w_pre,
            "h": data["height"][i] / h_pre,
        })
    return out


def find_text(
    image_bgr: np.ndarray, query_keywords: list[str],
    min_word_len: int = 3,
    crops: list[tuple[float, float, float, float]] | None = None,
) -> list[OCRHit]:
    """Find OCR words matching any of ``query_keywords``.

    Runs tesseract over both the full image and a small set of
    high-priority crops (so small text in sidebars is still picked
    up). Tries multiple PSM and scale combinations per region.
    """
    if not _HAVE_TESSERACT:
        return []
    if image_bgr.ndim != 3:
        return []
    if not query_keywords:
        return []

    h_orig, w_orig = image_bgr.shape[:2]
    queries_lower = [q.lower() for q in query_keywords]

    # Default crop set: full image + 4 quadrants + bottom strip.
    default_crops: list[tuple[float, float, float, float]] = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.45, 1.0),    # left third (sidebars)
        (0.0, 0.50, 0.45, 1.0),   # bottom-left
        (0.55, 0.0, 1.0, 0.5),    # top-right
        (0.0, 0.65, 1.0, 1.0),    # bottom strip (footers)
    ]
    crop_list = crops if crops is not None else default_crops

    # Multi-pass scan: try both polarities (dark-on-light + white-on-dark)
    # at multiple scales/psm. Tesseract is sensitive to all three.
    passes: list[tuple[int, int, bool]] = [
        (3, 11, False),  # sparse text, dark-on-light
        (3, 11, True),   # sparse text, light-on-dark (Reddit dark mode!)
        (4, 6, False),   # block text, dark-on-light
        (4, 6, True),    # block text, light-on-dark
        (5, 11, True),   # extra-large scale for tiny sidebar text
    ]

    hits: list[OCRHit] = []
    seen_keys: set[tuple[int, int, int, int, str]] = set()

    for x0f, y0f, x1f, y1f in crop_list:
        x0 = max(0, int(x0f * w_orig))
        y0 = max(0, int(y0f * h_orig))
        x1 = min(w_orig, int(x1f * w_orig))
        y1 = min(h_orig, int(y1f * h_orig))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        crop = image_bgr[y0:y1, x0:x1]
        crop_w = x1 - x0
        crop_h = y1 - y0

        for scale, psm, invert in passes:
            words = _ocr_words(crop, scale=scale, psm=psm, invert=invert)
            for rec in words:
                w = rec["word"]
                if len(w) < min_word_len:
                    continue
                if rec["conf"] < 0:
                    continue
                normalized_word = re.sub(r"[^a-z0-9]", "", w.lower())
                for q in queries_lower:
                    normalized_q = re.sub(r"[^a-z0-9]", "", q)
                    if not normalized_q:
                        continue
                    if normalized_q in normalized_word:
                        # Map crop fractions back to whole-image fractions.
                        wx = (rec["x"] * crop_w + x0) / w_orig
                        wy = (rec["y"] * crop_h + y0) / h_orig
                        ww = rec["w"] * crop_w / w_orig
                        wh = rec["h"] * crop_h / h_orig
                        key = (
                            int(wx * 1000), int(wy * 1000),
                            int(ww * 1000), int(wh * 1000),
                            w.lower(),
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        hits.append(OCRHit(
                            x_pct=wx + ww / 2,
                            y_pct=wy + wh / 2,
                            text=w,
                            confidence=rec["conf"],
                            bbox=(
                                int(wx * w_orig), int(wy * h_orig),
                                int(ww * w_orig), int(wh * h_orig),
                            ),
                        ))
                        break  # next word
    hits.sort(key=lambda h: -h.confidence)
    return hits


def annotate_ocr_hit(image_bgr: np.ndarray, hit: OCRHit) -> np.ndarray:
    """Annotate an OCR hit on the image for debugging."""
    out = image_bgr.copy()
    x, y, w, h = hit.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        out, f"OCR:{hit.text}({hit.confidence:.0f})",
        (x, max(0, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )
    h_img, w_img = out.shape[:2]
    cx = int(hit.x_pct * w_img)
    cy = int(hit.y_pct * h_img)
    cv2.circle(out, (cx, cy), 8, (0, 200, 255), 2)
    return out
