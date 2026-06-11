"""Synthetic stock-macOS-arrow cursor templates.

macOS draws the SAME arrow bitmap on every machine — black body,
white outline, hotspot at the tip. Unlike Linux (themeable, per-user)
we can render it offline and ship it, which makes template matching
work against an untouched remote Mac with zero target-side setup.

The templates are rendered at several heights because the cursor's
size in a webcam frame depends on how much of the frame the target
screen fills (and on the target's pointer-size accessibility
setting). The matcher tries all scales and keeps the best score.
"""

from __future__ import annotations

import cv2
import numpy as np

from handsneyes.platforms.base import CursorTemplate

# Stock macOS arrow polygon, normalised to height 1.0 (y down, x in
# the same unit so the arrow is ~0.62 wide). Traced from the standard
# artwork: tip → left edge down → notch → tail barb (out, across,
# back in) → right wing → tip.
_ARROW_POLY = np.array(
    [
        (0.000, 0.000),
        (0.000, 0.871),
        (0.211, 0.680),
        (0.355, 1.000),
        (0.494, 0.938),
        (0.345, 0.622),
        (0.622, 0.610),
    ],
    dtype=np.float32,
)

_BODY_VAL = 20.0      # near-black arrow body
_OUTLINE_VAL = 235.0  # white outline

# Heights (px) the matcher will try. Geometric-ish ladder covering a
# 1080p webcam frame where the target screen fills 40-100% of the
# view and the target's pointer size is default..large.
DEFAULT_HEIGHTS_PX = (10, 13, 17, 22, 28, 36, 46)


def render_arrow_template(height_px: int) -> CursorTemplate:
    """Render the stock arrow at ``height_px`` as a match-ready
    template: grayscale float32, zero-meaned over the cursor mask,
    softly blurred inside the mask to approximate webcam optics."""
    h = int(height_px)
    outline_t = max(1, int(round(h * 0.09)))
    pad = outline_t + 2
    pts = np.round(_ARROW_POLY * h).astype(np.int32) + pad
    canvas_h = h + 2 * pad
    canvas_w = int(round(0.622 * h)) + 2 * pad

    body = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(body, [pts], 255)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * outline_t + 1, 2 * outline_t + 1),
    )
    dilated = cv2.dilate(body, k)
    outline = cv2.bitwise_and(dilated, cv2.bitwise_not(body))

    img = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    img[outline > 0] = _OUTLINE_VAL
    img[body > 0] = _BODY_VAL
    mask = (dilated > 0).astype(np.float32)

    # Soft edges via normalised convolution (blur only inside the
    # mask so background pixels never contaminate the template).
    sigma = max(0.6, h / 30.0)
    blurred_img = cv2.GaussianBlur(img * mask, (0, 0), sigma)
    blurred_msk = cv2.GaussianBlur(mask, (0, 0), sigma)
    valid = blurred_msk > 0.35
    img = np.zeros_like(img)
    img[valid] = blurred_img[valid] / blurred_msk[valid]
    mask = valid.astype(np.float32)

    # Zero-mean over the mask: masked TM_CCORR_NORMED then scores ~0
    # on any uniform background patch instead of rewarding brightness.
    m = mask > 0
    img[m] -= float(img[m].mean())
    img[~m] = 0.0

    return CursorTemplate(
        name=f"macos-arrow-{h}",
        image=img,
        mask=mask,
        hotspot_px=(float(pad), float(pad)),
    )


# NOTE: an I-beam template ladder was prototyped and removed after
# live validation against a real remote-Mac HDMI frame: glyph-like
# shapes false-positive on terminal text ('|', quotes) in whole-
# frame searches, and the synthetic render never beat the arrow
# template even at the true I-beam position (0.32 vs 0.44, with a
# 9 px position bias). The arrow template empirically matches the
# real I-beam well enough to localize it. Don't re-add without
# validating against real frames first.

_CACHE: list[CursorTemplate] | None = None


def stock_cursor_templates() -> list[CursorTemplate]:
    global _CACHE
    if _CACHE is None:
        _CACHE = [render_arrow_template(h) for h in DEFAULT_HEIGHTS_PX]
    return _CACHE
