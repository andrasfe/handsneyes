"""Stock-cursor template rendering + multiscale matching.

Validates the zero-target-setup macOS detection path: synthetic
arrow templates shipped by the adapter, matched against a webcam-
simulated frame (blur + sensor noise + off-ladder cursor scale).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from handsneyes.core.vision.cursor_finder import (
    find_cursor_template_multiscale,
)
from handsneyes.platforms.macos import MacOSAdapter
from handsneyes.platforms.macos.cursor_templates import (
    DEFAULT_HEIGHTS_PX,
    render_arrow_template,
    stock_cursor_templates,
)

RNG = np.random.default_rng(42)


def _make_desktop(w: int = 1280, h: int = 720) -> np.ndarray:
    img = np.full((h, w), 210, np.uint8)
    for _ in range(40):
        x0, y0 = int(RNG.integers(0, w - 100)), int(RNG.integers(0, h - 80))
        ww, hh = int(RNG.integers(40, 400)), int(RNG.integers(30, 250))
        cv2.rectangle(
            img, (x0, y0), (min(w - 1, x0 + ww), min(h - 1, y0 + hh)),
            int(RNG.integers(30, 250)), -1,
        )
    for _ in range(150):
        x0, y0 = int(RNG.integers(0, w - 60)), int(RNG.integers(0, h - 10))
        cv2.line(
            img, (x0, y0), (x0 + int(RNG.integers(10, 60)), y0),
            int(RNG.integers(0, 255)), 1,
        )
    return img


def _stamp_cursor(
    img: np.ndarray, cx: int, cy: int, height: int,
) -> np.ndarray:
    """Draw a fresh arrow render (absolute grays, hotspot at cx,cy)."""
    t = render_arrow_template(height)
    m = t.mask > 0
    body = t.image.copy()
    body[m] = t.image[m] - t.image[m].min()
    body[m] = body[m] / max(1e-6, float(body[m].max())) * 215 + 20
    hx, hy = t.hotspot_px
    x0, y0 = int(cx - hx), int(cy - hy)
    th, tw = body.shape
    roi = img[y0:y0 + th, x0:x0 + tw].astype(np.float32)
    roi[m] = body[m]
    img[y0:y0 + th, x0:x0 + tw] = roi.astype(np.uint8)
    return img


def _webcam_sim(img: np.ndarray) -> np.ndarray:
    sim = cv2.GaussianBlur(img, (0, 0), 1.1)
    noise = RNG.normal(0, 4, sim.shape)
    return np.clip(sim.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def test_render_arrow_template_well_formed() -> None:
    for h in DEFAULT_HEIGHTS_PX:
        t = render_arrow_template(h)
        assert t.image.dtype == np.float32
        assert t.mask.dtype == np.float32
        assert t.image.shape == t.mask.shape
        m = t.mask > 0
        assert m.sum() > 0
        # Zero-meaned over the mask, zero outside it.
        assert abs(float(t.image[m].mean())) < 1e-3
        assert float(np.abs(t.image[~m]).max()) == 0.0
        # Hotspot inside the patch.
        hx, hy = t.hotspot_px
        assert 0 <= hx < t.image.shape[1]
        assert 0 <= hy < t.image.shape[0]


def test_stock_cursor_templates_cached() -> None:
    a = stock_cursor_templates()
    b = stock_cursor_templates()
    assert a is b
    assert len(a) == len(DEFAULT_HEIGHTS_PX)


def test_macos_adapter_ships_templates() -> None:
    templates = MacOSAdapter().cursor_templates()
    assert templates
    assert all(t.name.startswith("macos-arrow-") for t in templates)


@pytest.mark.parametrize("height", [14, 21, 31, 42])
def test_multiscale_finds_off_ladder_cursor(height: int) -> None:
    img = _make_desktop()
    cx, cy = int(RNG.integers(80, 1200)), int(RNG.integers(80, 640))
    img = _stamp_cursor(img, cx, cy, height)
    sim = _webcam_sim(img)
    hit = find_cursor_template_multiscale(
        sim, stock_cursor_templates(),
        search_center_pct=(cx / 1280, cy / 720),
        search_radius_pct=0.10,
    )
    assert hit is not None
    err = np.hypot(hit.x_pct * 1280 - cx, hit.y_pct * 720 - cy)
    assert err < 6.0, f"hotspot error {err:.1f}px (h={height})"


def test_multiscale_whole_frame_search() -> None:
    img = _make_desktop()
    cx, cy = 950, 530
    img = _stamp_cursor(img, cx, cy, 24)
    sim = _webcam_sim(img)
    hit = find_cursor_template_multiscale(sim, stock_cursor_templates())
    assert hit is not None
    err = np.hypot(hit.x_pct * 1280 - cx, hit.y_pct * 720 - cy)
    assert err < 6.0


def test_multiscale_no_false_positive_on_cursor_free_frame() -> None:
    for _ in range(5):
        sim = _webcam_sim(_make_desktop())
        hit = find_cursor_template_multiscale(
            sim, stock_cursor_templates(),
            search_center_pct=(0.5, 0.5), search_radius_pct=0.10,
        )
        assert hit is None


def test_multiscale_handles_empty_inputs() -> None:
    assert find_cursor_template_multiscale(None, stock_cursor_templates()) is None
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert find_cursor_template_multiscale(frame, []) is None
