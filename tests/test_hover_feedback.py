"""Hover-feedback measurement (cursor-free click confirmation)."""

from __future__ import annotations

import cv2
import numpy as np

from handsneyes.core.vision.cursor_finder import measure_hover_feedback

W, H = 1280, 720
TARGET = (0.50, 0.50)
BTN = (int(0.50 * W) - 60, int(0.50 * H) - 20, 120, 40)  # x, y, w, h
CURSOR_OFF = (0.05, 0.05)  # wiggle delta in image pct
RNG = np.random.default_rng(5)


def _frame(
    *, hover: bool, cursor_at_target: bool, seed_noise: bool = True,
) -> np.ndarray:
    img = np.full((H, W), 200, np.uint8)
    x, y, bw, bh = BTN
    cv2.rectangle(
        img, (x, y), (x + bw, y + bh), 160 if hover else 195, -1,
    )
    if cursor_at_target:
        cx, cy = int(TARGET[0] * W), int(TARGET[1] * H)
    else:
        cx = int((TARGET[0] + CURSOR_OFF[0]) * W)
        cy = int((TARGET[1] + CURSOR_OFF[1]) * H)
    cv2.fillPoly(
        img,
        [np.array(
            [[cx, cy], [cx, cy + 18], [cx + 11, cy + 12]], np.int32,
        )],
        30,
    )
    if seed_noise:
        img = np.clip(
            img.astype(np.float32) + RNG.normal(0, 1.0, img.shape),
            0, 255,
        ).astype(np.uint8)
    return img


def test_highlighting_button_yields_true() -> None:
    on1 = _frame(hover=True, cursor_at_target=True)
    off = _frame(hover=False, cursor_at_target=False)
    on2 = _frame(hover=True, cursor_at_target=True)
    assert measure_hover_feedback(
        on1, off, on2, target_pct=TARGET, cursor_footprint_px=150.0,
    ) is True


def test_non_highlighting_target_yields_false() -> None:
    on1 = _frame(hover=False, cursor_at_target=True)
    off = _frame(hover=False, cursor_at_target=False)
    on2 = _frame(hover=False, cursor_at_target=True)
    assert measure_hover_feedback(
        on1, off, on2, target_pct=TARGET, cursor_footprint_px=150.0,
    ) is False


def test_one_sided_repaint_yields_false() -> None:
    # A background repaint in only ONE interval (e.g. notification
    # popping) must not be mistaken for hover feedback — the signal
    # is min(out, back).
    on1 = _frame(hover=False, cursor_at_target=True)
    off = _frame(hover=True, cursor_at_target=False)   # changed once
    on2 = _frame(hover=True, cursor_at_target=True)    # stays changed
    assert measure_hover_feedback(
        on1, off, on2, target_pct=TARGET, cursor_footprint_px=150.0,
    ) is False


def test_animating_region_yields_none() -> None:
    def noisy() -> np.ndarray:
        img = _frame(hover=False, cursor_at_target=True, seed_noise=False)
        x, y, bw, bh = BTN
        patch = RNG.integers(0, 255, (bh * 3, bw * 2), dtype=np.uint8)
        img[y - bh:y + 2 * bh, x - bw // 2:x + bw + bw // 2] = patch
        return img

    assert measure_hover_feedback(
        noisy(), noisy(), noisy(),
        target_pct=TARGET, cursor_footprint_px=150.0,
    ) is None


def test_shape_mismatch_yields_none() -> None:
    a = np.zeros((720, 1280), np.uint8)
    b = np.zeros((480, 640), np.uint8)
    assert measure_hover_feedback(
        a, b, a, target_pct=TARGET,
    ) is None
