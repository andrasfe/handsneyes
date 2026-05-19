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
    enhance_for_ocr,
    enhance_for_screen,
    find_cursor_hsv,
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
