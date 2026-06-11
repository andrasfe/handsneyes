"""Screen-quad detection + perspective rectification."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
import pytest

from handsneyes.core.capture.base import CapturedFrame, CaptureSource
from handsneyes.core.capture.rectified import RectifiedCapture
from handsneyes.core.vision.screen_geometry import (
    ScreenRectifier,
    detect_screen_quad,
    order_corners,
)

RNG = np.random.default_rng(11)

# Screen quad in the synthetic 1280x720 "webcam frame" — slightly
# rotated and keystoned, like a webcam on a tripod next to the desk.
QUAD = np.array(
    [[180, 90], [1110, 130], [1060, 640], [140, 580]], dtype=np.float32,
)


def _make_screen_content(w: int = 960, h: int = 540) -> np.ndarray:
    """A bright 'desktop' with text-like noise + a red marker."""
    img = np.full((h, w, 3), 205, np.uint8)
    for _ in range(120):
        x0, y0 = int(RNG.integers(0, w - 80)), int(RNG.integers(0, h - 10))
        cv2.line(
            img, (x0, y0), (x0 + int(RNG.integers(20, 80)), y0),
            (60, 60, 60), 1,
        )
    return img


def _make_scene(
    marker_pct: tuple[float, float] | None = None,
) -> np.ndarray:
    """Dark noisy room with the bright screen warped in at QUAD."""
    room = np.clip(
        RNG.normal(28, 6, (720, 1280, 3)), 0, 255,
    ).astype(np.uint8)
    screen = _make_screen_content()
    if marker_pct is not None:
        mx = int(marker_pct[0] * screen.shape[1])
        my = int(marker_pct[1] * screen.shape[0])
        cv2.circle(screen, (mx, my), 6, (0, 0, 255), -1)
    sh, sw = screen.shape[:2]
    src = np.array(
        [[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, QUAD)
    warped = cv2.warpPerspective(screen, H, (1280, 720))
    mask = cv2.warpPerspective(
        np.full((sh, sw), 255, np.uint8), H, (1280, 720),
    )
    room[mask > 0] = warped[mask > 0]
    return room


def test_order_corners_orders_any_permutation() -> None:
    expected = np.array(
        [[10, 10], [90, 12], [88, 70], [8, 72]], dtype=np.float32,
    )
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        out = order_corners(expected[perm])
        assert np.allclose(out, expected)


def test_detect_screen_quad_finds_synthetic_screen() -> None:
    quad = detect_screen_quad(_make_scene())
    assert quad is not None
    err = np.linalg.norm(quad - QUAD, axis=1)
    assert err.max() < 8.0, f"corner errors {err}"


def test_detect_screen_quad_none_on_dark_room() -> None:
    room = np.clip(
        RNG.normal(28, 6, (720, 1280, 3)), 0, 255,
    ).astype(np.uint8)
    assert detect_screen_quad(room) is None


def test_rectifier_maps_screen_point_to_true_pct() -> None:
    marker = (0.31, 0.68)
    scene = _make_scene(marker_pct=marker)
    quad = detect_screen_quad(scene)
    assert quad is not None
    rect = ScreenRectifier.from_quad(
        quad, (1280, 720), aspect_hint=(1920, 1080),
    )
    out = rect.rectify(scene)
    oh, ow = out.shape[:2]
    assert (ow, oh) == rect.out_size
    assert abs(ow / oh - 1920 / 1080) < 0.01
    # Find the red marker in the rectified frame.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    M = cv2.moments(cv2.bitwise_or(m1, m2), binaryImage=True)
    assert M["m00"] > 0
    found = (M["m10"] / M["m00"] / ow, M["m01"] / M["m00"] / oh)
    err = np.hypot(found[0] - marker[0], found[1] - marker[1])
    assert err < 0.012, f"marker pct error {err:.4f}"


def test_rectifier_persistence_round_trip(tmp_path) -> None:
    rect = ScreenRectifier.from_quad(QUAD, (1280, 720))
    p = tmp_path / "rectify.json"
    rect.save(p)
    loaded = ScreenRectifier.load(p, expect_frame_size=(1280, 720))
    assert loaded is not None
    assert np.allclose(loaded.corners, rect.corners)
    assert loaded.out_size == rect.out_size
    # Frame-size mismatch → refuse.
    assert ScreenRectifier.load(p, expect_frame_size=(640, 480)) is None
    # Missing / corrupt → None, never raises.
    assert ScreenRectifier.load(tmp_path / "nope.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert ScreenRectifier.load(tmp_path / "bad.json") is None


class _FakeCapture(CaptureSource):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__()
        self.frame = frame

    async def open(self) -> None:
        self._is_open = True

    async def close(self) -> None:
        self._is_open = False

    async def capture_frame(self) -> CapturedFrame:
        self._frame_counter += 1
        return CapturedFrame(
            image=self.frame, frame_number=self._frame_counter,
        )


def test_rectified_capture_calibrates_and_warps(tmp_path) -> None:
    scene = _make_scene()
    cap = RectifiedCapture(
        _FakeCapture(scene),
        aspect_hint=(1920, 1080),
        cache_path=tmp_path / "cal.json",
    )

    async def run():
        await cap.open()
        f = await cap.capture_frame()
        await cap.close()
        return f

    frame = asyncio.run(run())
    assert cap.rectifier is not None
    assert frame.image.shape[:2][::-1] == cap.rectifier.out_size
    assert (tmp_path / "cal.json").exists()


def test_rectified_capture_passthrough_without_quad(tmp_path) -> None:
    room = np.clip(
        RNG.normal(28, 6, (720, 1280, 3)), 0, 255,
    ).astype(np.uint8)
    cap = RectifiedCapture(
        _FakeCapture(room), cache_path=tmp_path / "cal.json",
    )

    async def run():
        await cap.open()
        f = await cap.capture_frame()
        await cap.close()
        return f

    frame = asyncio.run(run())
    assert cap.rectifier is None
    assert frame.image.shape == room.shape


def test_rectified_capture_uses_cache_when_screen_dark(tmp_path) -> None:
    cache = tmp_path / "cal.json"
    ScreenRectifier.from_quad(QUAD, (1280, 720)).save(cache)
    room = np.clip(
        RNG.normal(28, 6, (720, 1280, 3)), 0, 255,
    ).astype(np.uint8)
    cap = RectifiedCapture(_FakeCapture(room), cache_path=cache)
    asyncio.run(cap.open())
    assert cap.rectifier is not None


def test_targets_loader_parses_rectify_flag(tmp_path) -> None:
    from handsneyes.targets import TargetRegistry

    toml = tmp_path / "targets.toml"
    toml.write_text(
        '[[target]]\nname = "rmac"\nplatform = "macos"\nrectify = true\n'
        '\n[[target]]\nname = "plain"\nplatform = "macos"\n',
        encoding="utf-8",
    )
    reg = TargetRegistry.from_file(toml)
    assert reg.targets["rmac"].rectify is True
    assert reg.targets["plain"].rectify is False
