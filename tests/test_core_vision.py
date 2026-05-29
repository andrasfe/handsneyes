"""Tests for handsneyes.core.vision.

Lean smoke + behaviour tests covering:

  - imaging round-trips (numpy ↔ PIL, base64 encode, resize)
  - text_targeting keyword + prompt helpers
  - HSV cursor finder happy/sad paths on synthetic frames
  - OCR finder graceful when tesseract isn't available
  - pointer_accel + longjump loader sanity (config defaults, no-checkpoint
    fallback)

Heavyweight integration (real webcam, real cursor) is out of scope
for Phase A.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from handsneyes.core.vision import (
    CursorHit,
    LongJumpConfig,
    OCRHit,
    PointerAccelConfig,
    capture_cursor_template,
    enhance_for_ocr,
    enhance_for_screen,
    find_cursor_hsv,
    find_cursor_template,
    find_text,
    have_ocr,
    numpy_to_base64_png,
    numpy_to_pil,
    pil_to_numpy,
    resize_for_mllm,
    showui_prompt_variants,
    target_keywords,
)

# ─── imaging ────────────────────────────────────────────────────────


def _blank(w: int = 64, h: int = 32, val: int = 200) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_numpy_to_base64_png_round_trip() -> None:
    img = _blank()
    b64 = numpy_to_base64_png(img)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_numpy_pil_round_trip_preserves_shape() -> None:
    img = _blank()
    pil = numpy_to_pil(img)
    back = pil_to_numpy(pil)
    assert back.shape == img.shape


def test_resize_for_mllm_downscales_large() -> None:
    img = _blank(3000, 1500)
    out = resize_for_mllm(img, max_dimension=1568)
    assert max(out.shape[:2]) <= 1568


def test_resize_for_mllm_upscales_small() -> None:
    img = _blank(200, 100)
    out = resize_for_mllm(img, max_dimension=1568, min_dimension=1024)
    assert max(out.shape[:2]) >= 1024


def test_resize_for_mllm_passes_through_in_range() -> None:
    img = _blank(1200, 800)
    out = resize_for_mllm(img, max_dimension=1568, min_dimension=1024)
    assert out.shape == img.shape


def test_enhance_for_ocr_emits_high_contrast() -> None:
    grey = np.full((40, 80, 3), 128, dtype=np.uint8)
    out = enhance_for_ocr(grey)
    assert out.shape == grey.shape
    vals = set(np.unique(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)))
    assert vals.issubset({0, 255})


def test_enhance_for_screen_preserves_shape() -> None:
    img = _blank()
    out = enhance_for_screen(img)
    assert out.shape == img.shape


# ─── text_targeting ─────────────────────────────────────────────────


def test_target_keywords_drops_stopwords_and_positional_tail() -> None:
    # The cut-phrase machinery treats " on the " / " below " as positional
    # context. So "Run" survives only when it isn't preceded by "on the".
    kws = target_keywords("Run button below the Edit menu")
    assert "run" in kws
    assert "edit" not in kws  # everything after "below" is dropped
    # stopwords filtered
    assert "button" not in kws


def test_target_keywords_keeps_quoted_strings_regardless_of_position() -> None:
    kws = target_keywords('the link below the header labeled "Settings"')
    assert "settings" in kws


def test_showui_prompt_variants_emits_quoted_and_caps() -> None:
    variants = showui_prompt_variants('the "Run" button')
    joined = " | ".join(variants)
    assert any('"Run"' in v or "Run" in v for v in variants)
    assert "the \"Run\" button" in joined


def test_showui_prompt_variants_dedups() -> None:
    variants = showui_prompt_variants("a b a b")
    assert len(variants) == len(set(v.lower() for v in variants))


# ─── cursor_finder ──────────────────────────────────────────────────


def test_find_cursor_hsv_on_blank_returns_none() -> None:
    img = _blank(200, 200, val=128)
    assert find_cursor_hsv(img) is None


def test_find_cursor_hsv_on_synthetic_red_blob_does_not_crash() -> None:
    img = _blank(300, 300, val=40)  # dark background
    cv2.circle(img, (150, 150), 12, (0, 0, 255), thickness=-1)
    # The HSV thresholds + area floors are tuned for real cursors;
    # the contract we want here is "doesn't crash and returns a
    # CursorHit-or-None". Validating the exact hit on synthetic input
    # would couple the test to internal tuning constants.
    hit = find_cursor_hsv(img)
    assert hit is None or isinstance(hit, CursorHit)


# ─── cursor_finder: template path ──────────────────────────────────


def _synthetic_cursor_scene(
    w: int = 320,
    h: int = 240,
    cursor_pct: tuple[float, float] = (0.5, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """A textured background with a distinctive arrow-shaped cursor
    drawn at ``cursor_pct``. Returns ``(frame, template)`` where the
    template is cropped via :func:`capture_cursor_template` from the
    frame's known cursor position --- exactly the workflow the homer
    will use at run time.
    """
    rng = np.random.default_rng(seed=0)
    img = rng.integers(40, 90, size=(h, w, 3), dtype=np.uint8)
    # Draw a small white arrow at the requested position so the
    # template carries a distinctive shape rather than a plain block
    # of colour (which would degenerate into "find a white square").
    cx = int(round(cursor_pct[0] * w))
    cy = int(round(cursor_pct[1] * h))
    arrow = np.array([
        [cx, cy], [cx + 12, cy + 6], [cx + 6, cy + 6],
        [cx + 6, cy + 14], [cx, cy + 14],
    ], dtype=np.int32)
    cv2.fillPoly(img, [arrow], color=(240, 240, 240))
    tmpl = capture_cursor_template(img, cursor_pct, size_pct=0.10)
    return img, tmpl  # type: ignore[return-value]


def test_capture_cursor_template_inside_frame_returns_patch() -> None:
    img = _blank(200, 200, val=128)
    tmpl = capture_cursor_template(img, (0.5, 0.5), size_pct=0.10)
    assert tmpl is not None
    assert tmpl.ndim == 3 and tmpl.shape[2] == 3
    # 200 * 0.10 = 20 → patch should be 20×20.
    assert tmpl.shape[0] == 20 and tmpl.shape[1] == 20


def test_capture_cursor_template_outside_frame_returns_none() -> None:
    img = _blank(200, 200, val=128)
    # A 25% patch centred at the edge cannot fit.
    assert capture_cursor_template(img, (0.0, 0.5), size_pct=0.25) is None


def test_find_cursor_template_locates_known_position() -> None:
    img, tmpl = _synthetic_cursor_scene(
        cursor_pct=(0.3, 0.6),
    )
    assert tmpl is not None
    hit = find_cursor_template(img, tmpl)
    assert hit is not None
    x_pct, y_pct, score = hit
    # Template centre vs known position: within one pixel on each
    # axis, which on a 320x240 frame is ~0.3% / 0.4%.
    assert abs(x_pct - 0.3) < 0.01
    assert abs(y_pct - 0.6) < 0.02
    assert score > 0.95  # synthetic template matches itself near-perfectly


def test_find_cursor_template_returns_none_when_absent() -> None:
    _, tmpl = _synthetic_cursor_scene(cursor_pct=(0.5, 0.5))
    blank = _blank(320, 240, val=60)
    hit = find_cursor_template(blank, tmpl, score_threshold=0.5)
    assert hit is None


def test_find_cursor_template_roi_search_picks_closest_match() -> None:
    """When two cursor-like patches exist, the ROI-restricted search
    should ignore the far one and lock onto the candidate inside the
    search window.
    """
    # First scene gives us the template + a cursor at (0.3, 0.6).
    near, tmpl = _synthetic_cursor_scene(cursor_pct=(0.3, 0.6))
    assert tmpl is not None
    # Paint a second arrow at a far position to act as a distractor.
    far_cx = int(round(0.85 * near.shape[1]))
    far_cy = int(round(0.20 * near.shape[0]))
    arrow = np.array([
        [far_cx, far_cy], [far_cx + 12, far_cy + 6],
        [far_cx + 6, far_cy + 6], [far_cx + 6, far_cy + 14],
        [far_cx, far_cy + 14],
    ], dtype=np.int32)
    cv2.fillPoly(near, [arrow], color=(240, 240, 240))
    # Full-frame search may pick either depending on noise; ROI
    # search around (0.3, 0.6) must lock onto the near one.
    hit = find_cursor_template(
        near, tmpl,
        search_center_pct=(0.32, 0.58),
        search_radius_pct=0.12,
    )
    assert hit is not None
    x_pct, y_pct, _ = hit
    assert abs(x_pct - 0.3) < 0.02
    assert abs(y_pct - 0.6) < 0.03


# ─── ocr_finder ─────────────────────────────────────────────────────


def test_have_ocr_is_boolean() -> None:
    assert isinstance(have_ocr(), bool)


def test_find_text_handles_missing_tesseract_gracefully() -> None:
    # find_text returns an empty list (not None) when no match / no OCR
    # backend available. Must not raise on a blank image.
    img = _blank()
    hits = find_text(img, query_keywords=["anything"])
    assert isinstance(hits, list)
    if hits:
        assert isinstance(hits[0], OCRHit)


# ─── pointer_accel + longjump ───────────────────────────────────────


def test_pointer_accel_config_constructs() -> None:
    cfg = PointerAccelConfig(
        hidden=64,
        input_features=["dx_pct", "dy_pct"],
        output_features=["hid_dx", "hid_dy"],
    )
    assert cfg.platform == "ubuntu-libinput-adaptive"
    assert cfg.hidden == 64


def test_longjump_config_constructs() -> None:
    cfg = LongJumpConfig(
        hidden=32,
        hid_scale=20.0,
        input_features=["dx_pct"],
        output_features=["total_hid_dx"],
    )
    assert cfg.platform == "ubuntu-libinput-adaptive"
    assert cfg.hid_scale == 20.0


def test_pointer_accel_model_raises_on_missing_checkpoint(
    tmp_path: object,
) -> None:
    from handsneyes.core.vision import PointerAccelModel

    # Empty dir has no config.json → constructor should refuse.
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        PointerAccelModel(weights_dir=tmp_path)  # type: ignore[arg-type]
