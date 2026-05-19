# ruff: noqa: N806
#   M = cv2.moments(...) and similar mathematical conventions are
#   intentionally uppercase in cv2/numpy code.
"""HSV-based cursor finder.

Premise: classical computer vision can locate the mouse cursor in a
webcam frame in milliseconds — but only if the cursor itself stands out.

**Ubuntu / GNOME setup (one-time, on the target machine):**

    sudo apt install -y xcursor-themes
    gsettings set org.gnome.desktop.interface cursor-theme 'redglass'
    gsettings set org.gnome.desktop.interface cursor-size 96
    # log out / log in (or open a fresh app) for the change to take effect

``redglass`` is a saturated-red X11 cursor that ships with the
``xcursor-themes`` package on Ubuntu. At size 96 it's roughly 50 px on
a webcam viewing a 1080p screen — easy to find by HSV thresholding for
red. No models, no diff, no per-step token spend.

If detection misses (cursor not set up, or unusual webcam colour cast),
this returns ``None`` and ``setup_instructions()`` prints what to do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Red wraps in HSV; cover both ends. Saturation/value floors keep us
# from latching onto pinkish skin tones, brown text, etc.
RED_LO_A = np.array([0,   140, 110], dtype=np.uint8)
RED_HI_A = np.array([10,  255, 255], dtype=np.uint8)
RED_LO_B = np.array([170, 140, 110], dtype=np.uint8)
RED_HI_B = np.array([180, 255, 255], dtype=np.uint8)


# Spatial constraints (image-percent / image-area).
# At cursor-size 96 on a 1080p screen filling ~70% of a 1280×720 webcam,
# the cursor occupies ~30–60 px → ~900–3600 px². Allow a wide range.
MIN_AREA_PCT = 0.00015   # ~140 px on a 1280×720 frame
MAX_AREA_PCT = 0.020     # ~18 000 px on the same frame


@dataclass
class CursorHit:
    x_pct: float
    y_pct: float
    area_pct: float
    confidence: float  # 0..1, blends size + colour purity


def find_cursor_hsv(image_bgr: np.ndarray) -> CursorHit | None:
    """Find the saturated-red ``redglass`` cursor in a BGR frame.

    Returns ``None`` if no plausible candidate is present.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    h, w = image_bgr.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red_a = cv2.inRange(hsv, RED_LO_A, RED_HI_A)
    red_b = cv2.inRange(hsv, RED_LO_B, RED_HI_B)
    red_mask = cv2.bitwise_or(red_a, red_b)

    # Clean small noise; close so the cursor's outline+body merge into
    # one blob even if the centre highlight breaks the red region.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel5)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    best: CursorHit | None = None
    for c in contours:
        area = cv2.contourArea(c)
        area_pct = area / img_area
        if area_pct < MIN_AREA_PCT:
            continue
        if area_pct > MAX_AREA_PCT:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Confidence: prefer mid-range blob sizes (cursor-sized).
        size_score = min(1.0, area_pct / 0.005)
        # Penalize blobs at extreme image edges (bezel/environment).
        edge_score = 1.0
        margin = 0.02
        if (cx / w < margin or cx / w > 1 - margin
                or cy / h < margin or cy / h > 1 - margin):
            edge_score = 0.3
        confidence = size_score * edge_score
        if best is None or confidence > best.confidence:
            best = CursorHit(
                x_pct=cx / w, y_pct=cy / h,
                area_pct=area_pct, confidence=confidence,
            )
    return best


# Wider red gates for the position-aware variant. The global
# ``find_cursor_hsv`` has to filter out skin tones and brown text
# from the entire frame; ``find_cursor_hsv_near`` only looks inside
# a small ROI around a known cursor position, so it can accept much
# weaker reds (low saturation from webcam perspective, glare, etc.)
# without false-positiving on environment.
WIDE_RED_LO_A = np.array([0,    60,  60], dtype=np.uint8)
WIDE_RED_HI_A = np.array([18,  255, 255], dtype=np.uint8)
WIDE_RED_LO_B = np.array([160,  60,  60], dtype=np.uint8)
WIDE_RED_HI_B = np.array([180, 255, 255], dtype=np.uint8)
# Much smaller floor too — the ROI is bounded so a 30-px-wide cursor
# is plenty.
WIDE_MIN_AREA_PCT = 0.00004   # ~80 px on a 1920x1080 frame


def find_cursor_hsv_near(
    image_bgr: np.ndarray,
    near_pct: tuple[float, float],
    max_dist_pct: float = 0.04,
) -> CursorHit | None:
    """Variant of :func:`find_cursor_hsv` that prefers a blob whose
    centroid is close to ``near_pct`` (an externally-known cursor
    position, e.g. from oscillation-variance detection).

    Solves the case where the globally highest-confidence red blob
    in the frame is a static UI accent at the screen edge, not the
    cursor. The cursor itself is somewhere we already know, so
    "biggest blob within ``max_dist_pct`` of that point" is provably
    the right one — and using its HSV centroid gives us a pixel-
    accurate position to feed the closed-loop servo, far better than
    the frame-diff fallback.

    Uses wider HSV thresholds and a lower minimum-area floor than
    the global finder because the position constraint already
    excludes the false positives that the strict thresholds existed
    to filter out. This way a desaturated webcam capture of the
    redglass cursor still registers.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    h, w = image_bgr.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red_a = cv2.inRange(hsv, WIDE_RED_LO_A, WIDE_RED_HI_A)
    red_b = cv2.inRange(hsv, WIDE_RED_LO_B, WIDE_RED_HI_B)
    red_mask = cv2.bitwise_or(red_a, red_b)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel5)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    best: CursorHit | None = None
    best_dist = float("inf")
    for c in contours:
        area = cv2.contourArea(c)
        area_pct = area / img_area
        if area_pct < WIDE_MIN_AREA_PCT:
            continue
        if area_pct > MAX_AREA_PCT:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x_pct = cx / w
        y_pct = cy / h
        dx = x_pct - near_pct[0]
        dy = y_pct - near_pct[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_dist_pct:
            continue
        # Among in-range blobs, prefer the closest (the cursor itself,
        # not a nearby UI fleck).
        if dist < best_dist:
            best_dist = dist
            size_score = min(1.0, area_pct / 0.005)
            best = CursorHit(
                x_pct=x_pct, y_pct=y_pct,
                area_pct=area_pct,
                # Confidence: combines size and proximity.
                confidence=size_score
                * max(0.0, 1.0 - dist / max_dist_pct),
            )
    return best


def find_cursor_hsv_motion(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    *,
    near_pct: tuple[float, float] | None = None,
    max_dist_pct: float | None = None,
    dilate_px: int = 4,
) -> CursorHit | None:
    """Find the cursor via differential red-mask between a pre-HID
    and post-HID frame.

    The cursor is the only red thing on the host that responds to
    our HID commands. Every other red region (UI accent, syntax-
    highlighted text, dialog icon, brand colour) is static. By
    masking only the pixels that BECAME red between the two frames,
    we cancel out all static red regardless of how much exists on
    screen.

    Algorithm:
      1. red_mask_pre, red_mask_post = HSV-threshold both frames.
      2. Dilate red_mask_pre by ``dilate_px`` so a ±dilate_px webcam
         jitter on a static blob still cancels.
      3. newly_red = red_mask_post AND NOT dilated_red_mask_pre.
      4. Largest contour in newly_red (passing area filters and the
         optional ``near_pct`` proximity gate) is the cursor.

    Optional ``near_pct`` + ``max_dist_pct`` add a position prior —
    useful when the homer knows roughly where the cursor should
    have landed after a known HID burst.

    Returns ``None`` if no blob survives filtering — pre-frame
    coverage was high enough that nothing was newly red (rare in
    practice; the cursor's pre-position becomes "newly NOT red" but
    its post-position is brand-new red).
    """
    if (pre_bgr.ndim != 3 or post_bgr.ndim != 3
            or pre_bgr.shape != post_bgr.shape):
        return None
    h, w = post_bgr.shape[:2]
    img_area = h * w

    def _red(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, WIDE_RED_LO_A, WIDE_RED_HI_A)
        m2 = cv2.inRange(hsv, WIDE_RED_LO_B, WIDE_RED_HI_B)
        return cv2.bitwise_or(m1, m2)

    mask_pre = _red(pre_bgr)
    mask_post = _red(post_bgr)

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1),
        )
        mask_pre_dil = cv2.dilate(mask_pre, k)
    else:
        mask_pre_dil = mask_pre

    newly_red = cv2.bitwise_and(mask_post, cv2.bitwise_not(mask_pre_dil))
    # Clean noise; close adjacent fragments into a single blob.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    newly_red = cv2.morphologyEx(newly_red, cv2.MORPH_OPEN, kernel3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    newly_red = cv2.morphologyEx(newly_red, cv2.MORPH_CLOSE, kernel5)

    contours, _ = cv2.findContours(
        newly_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    best: CursorHit | None = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        area_pct = area / img_area
        if area_pct < WIDE_MIN_AREA_PCT:
            continue
        if area_pct > MAX_AREA_PCT:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x_pct = cx / w
        y_pct = cy / h
        if near_pct is not None and max_dist_pct is not None:
            dx = x_pct - near_pct[0]
            dy = y_pct - near_pct[1]
            if (dx * dx + dy * dy) ** 0.5 > max_dist_pct:
                continue
        if area > best_area:
            best_area = area
            best = CursorHit(
                x_pct=x_pct, y_pct=y_pct,
                area_pct=area_pct,
                confidence=min(1.0, area_pct / 0.005),
            )
    return best


def find_cursor_hsv_motion_directed(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    *,
    cursor_pre_pct: tuple[float, float],
    expected_motion_pct: tuple[float, float],
    max_dist_pct: float = 0.08,
    dilate_px: int = 4,
    require_arrow_shape: bool = True,
    min_cos_similarity: float = 0.5,
) -> CursorHit | None:
    """Locate the cursor using the motion-diff red mask AND two
    additional priors:

      1. Direction match — we know what HID we sent, so the cursor's
         observed displacement should align with the expected motion
         vector. Each candidate's ``cos(observed, expected)`` must
         exceed ``min_cos_similarity``. This rejects "newly red"
         regions caused by side effects (a dialog or menu appearing
         after the click) since their apparent position relative to
         the previous cursor doesn't lie in the HID direction.
      2. Arrow-like shape — the redglass cursor is a tall asymmetric
         arrow (aspect ratio h/w roughly 1.2-2.5, fairly solid).
         Static UI red is usually round (close-button), wide (text
         highlight), or sparse (icon outline). Filtering by shape
         double-filters in addition to (1).

    ``cursor_pre_pct`` is the cursor's position BEFORE the HID
    (image-percent). ``expected_motion_pct`` is the predicted
    displacement (target_new - cursor_pre). Both are required;
    callers without them should use ``find_cursor_hsv_motion``
    instead.

    Returns the best-scoring candidate, or None if no newly-red
    blob satisfies all filters.
    """
    if (pre_bgr.ndim != 3 or post_bgr.ndim != 3
            or pre_bgr.shape != post_bgr.shape):
        return None
    h, w = post_bgr.shape[:2]
    img_area = h * w

    def _red(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, WIDE_RED_LO_A, WIDE_RED_HI_A)
        m2 = cv2.inRange(hsv, WIDE_RED_LO_B, WIDE_RED_HI_B)
        return cv2.bitwise_or(m1, m2)

    mask_pre = _red(pre_bgr)
    mask_post = _red(post_bgr)
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1),
        )
        mask_pre_dil = cv2.dilate(mask_pre, k)
    else:
        mask_pre_dil = mask_pre
    newly_red = cv2.bitwise_and(mask_post, cv2.bitwise_not(mask_pre_dil))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    newly_red = cv2.morphologyEx(newly_red, cv2.MORPH_OPEN, kernel3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    newly_red = cv2.morphologyEx(newly_red, cv2.MORPH_CLOSE, kernel5)
    contours, _ = cv2.findContours(
        newly_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    expected_new_pct = (
        cursor_pre_pct[0] + expected_motion_pct[0],
        cursor_pre_pct[1] + expected_motion_pct[1],
    )
    ex_mag = math.hypot(*expected_motion_pct)

    best: CursorHit | None = None
    best_score = -1.0
    best_diag = None
    for c in contours:
        area = cv2.contourArea(c)
        area_pct = area / img_area
        if area_pct < WIDE_MIN_AREA_PCT or area_pct > MAX_AREA_PCT:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x_pct = cx / w
        y_pct = cy / h
        # Position prior — must land near expected_new.
        d_to_expected = math.hypot(
            x_pct - expected_new_pct[0],
            y_pct - expected_new_pct[1],
        )
        if d_to_expected > max_dist_pct:
            continue
        # Shape prior — arrow aspect ratio + solidity.
        _, _, bw, bh = cv2.boundingRect(c)
        aspect = bh / max(1, bw)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)
        shape_ok = (
            0.9 <= aspect <= 3.0 and solidity >= 0.45
            if require_arrow_shape else True
        )
        if require_arrow_shape and not shape_ok:
            continue
        # Direction prior — observed displacement vs expected.
        obs_dx = x_pct - cursor_pre_pct[0]
        obs_dy = y_pct - cursor_pre_pct[1]
        obs_mag = math.hypot(obs_dx, obs_dy)
        if ex_mag > 1e-6 and obs_mag > 1e-6:
            cos_sim = (
                (obs_dx * expected_motion_pct[0]
                 + obs_dy * expected_motion_pct[1])
                / (obs_mag * ex_mag)
            )
        else:
            # Either the expected motion or observed motion is ~0;
            # direction is meaningless. Pass the gate but get no
            # direction bonus.
            cos_sim = 1.0
        if cos_sim < min_cos_similarity:
            continue
        # Composite score: closer to expected + better direction
        # match + larger blob (within the cursor-size band) + more
        # arrow-like.
        proximity_score = max(0.0, 1.0 - d_to_expected / max_dist_pct)
        size_score = min(1.0, area_pct / 0.005)
        aspect_score = 1.0 - abs(aspect - 1.7) / 1.7  # peaks at h/w=1.7
        score = (
            2.0 * cos_sim
            + 1.5 * proximity_score
            + 0.5 * size_score
            + 0.5 * max(0.0, aspect_score)
        )
        if score > best_score:
            best_score = score
            best = CursorHit(
                x_pct=x_pct, y_pct=y_pct,
                area_pct=area_pct,
                confidence=min(1.0, score / 4.5),
            )
            best_diag = (cos_sim, aspect, solidity, d_to_expected)
    if best is not None and best_diag is not None:
        import logging
        logging.getLogger(__name__).debug(
            "find_cursor_hsv_motion_directed: hit=(%.2f%%,%.2f%%) "
            "cos=%.2f aspect=%.2f solidity=%.2f d=%.3f score=%.2f",
            best.x_pct * 100, best.y_pct * 100,
            best_diag[0], best_diag[1], best_diag[2], best_diag[3],
            best_score,
        )
    return best


def annotate_cursor(
    image_bgr: np.ndarray, hit: CursorHit,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the cursor hit marked."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    cx = int(hit.x_pct * w)
    cy = int(hit.y_pct * h)
    cv2.circle(out, (cx, cy), 22, color, 2)
    cv2.line(out, (cx - 30, cy), (cx + 30, cy), color, 1)
    cv2.line(out, (cx, cy - 30), (cx, cy + 30), color, 1)
    cv2.putText(
        out, f"cursor a={hit.area_pct:.4f} c={hit.confidence:.2f}",
        (cx + 24, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )
    return out


def find_cursor_by_variance(
    frames: list[np.ndarray],
    variance_threshold: float = 8.0,
    min_active_pixels: int = 30,
    max_active_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """Find the cursor by computing the centroid of all high-variance
    pixels across frames captured during cursor oscillation.

    Premise: when we jiggle the cursor (send small ±moves and back),
    only the cursor's pixels change between frames. Static UI ≈ 0
    variance. The cursor leaves a small "trail" of high-variance
    pixels at each oscillation position. The centroid of that trail
    is approximately the cursor's start position (oscillation is
    symmetric, so the trajectory centres on where it started).

    Uses a fixed variance threshold (default 8) rather than a
    percentile — with ~hundreds of moving pixels in a multi-million
    pixel frame, the 99th percentile is 0 and the percentile-based
    cutoff would short-circuit. Cursor pixels have std ≈ 40–80
    across the oscillation, well above 8.

    Returns ``None`` if the variance signal is too weak
    (``< min_active_pixels``) or too broad (more than
    ``max_active_fraction`` of the image is "active" — likely
    something other than the cursor is animating).
    """
    if len(frames) < 3:
        return None
    if any(f.ndim != 2 for f in frames):
        frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            for f in frames
        ]
    h, w = frames[0].shape[:2]
    img_area = h * w
    arr = np.stack([f.astype(np.float32) for f in frames], axis=0)
    var = arr.std(axis=0)
    mask = (var > variance_threshold).astype(np.uint8) * 255
    # Open to drop isolated noise pixels; keep cursor outlines intact.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)

    active_pixels = int((mask > 0).sum())
    if active_pixels < min_active_pixels:
        return None
    if active_pixels > img_area * max_active_fraction:
        # Too many pixels are moving — something else is animating.
        return None

    # Centroid of the entire active mask. Robust to multiple
    # disconnected blobs (cursor visited several positions during the
    # jiggle): the geometric centre of all positions ≈ the original
    # cursor position when the oscillation pattern is symmetric.
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx / w, cy / h


def setup_instructions() -> str:
    """One-shot setup for the target Ubuntu machine's cursor.

    Print to the user when ``find_cursor_hsv`` keeps returning
    ``None`` so they can switch to the bright-red redglass theme.
    """
    return (
        "═══════════ CURSOR SETUP REQUIRED (Ubuntu) ═══════════\n"
        "On the TARGET machine, run:\n"
        "  sudo apt install -y xcursor-themes\n"
        "  gsettings set org.gnome.desktop.interface cursor-theme 'redglass'\n"
        "  gsettings set org.gnome.desktop.interface cursor-size 96\n"
        "Then log out and back in (or open a new app) so the cursor\n"
        "actually changes. Why: the homer locates the cursor by HSV\n"
        "thresholding for saturated red — redglass at size 96 is\n"
        "unmistakable on a webcam, no model calls needed.\n"
        "══════════════════════════════════════════════════════"
    )
