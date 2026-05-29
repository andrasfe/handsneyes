"""Camera-index autodetect for handsneyes targets.

The dev Mac typically presents several video-capture endpoints at
cv2's enumerated indices, mixing physical and virtual cameras:

  - the built-in FaceTime camera
  - one or more USB webcams (pointed at the operator, or at a
    physical monitor)
  - virtual cameras exposing screen-share, Sidecar, AirPlay
    receiver, or Continuity feeds (these appear as "cameras" to
    cv2 but carry pixel-perfect screencaps of another machine's
    desktop)

For a remote-control target the useful endpoint is either

  (a) a virtual camera feeding the remote machine's mirrored
      screen --- screencap-style, low temporal noise --- which is
      the typical setup when the remote Mac AirPlays or
      screen-shares to the dev Mac, or
  (b) a physical webcam pointed at the remote machine's monitor
      --- live feed, high temporal noise (sensor noise even when
      the scene is otherwise static),

and we explicitly do NOT want the dev Mac capturing its OWN screen
(useless: we would not see what we are about to control).

The autodetect picks an index in this order:

  1. Probe cv2 indices ``0..max_index``.
  2. Skip indices that fail to open or read frames.
  3. Skip indices whose frame matches the dev Mac's own desktop
     under a downsampled L1 comparison (dev-self-capture).
  4. Among the rest, prefer LOW temporal noise (the screen-share
     case the user's setup actually uses); fall back to live feeds
     when nothing screen-share-style is available.

The decision is logged at INFO so operators can see why a particular
index was selected without re-running the probe manually.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Thresholds calibrated against macOS probes: a real USB webcam
# produces per-pixel temporal std around 3+ (sensor noise even on a
# static scene); a screen-share or virtual camera with an idle
# desktop sits around 0.2. The 0.5 cutoff is comfortably between
# the two clusters.
_STATIC_TEMPORAL_STD = 0.5
# Downsampled L1 distance threshold: dev mac's own screen captured
# back through a virtual camera tends to land under 20; the remote
# Mac's screen (or a webcam feed) is comfortably higher.
_DEV_SELF_DIFF_MAX = 20.0


def _grab_dev_screen_thumb(size: int = 128) -> Optional[np.ndarray]:
    """Best-effort thumbnail of the dev mac's own desktop, used to
    veto cv2 indices that are just self-capturing the dev mac.

    Returns ``None`` on platforms / configurations where PIL
    ``ImageGrab`` is unavailable; the caller then skips the
    dev-self-capture veto and falls back to noise-based selection
    alone.
    """
    try:
        from PIL import ImageGrab
        ref = np.array(ImageGrab.grab(all_screens=False))
        ref_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
        return cv2.resize(
            ref_bgr, (size, size), interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
    except Exception as e:
        logger.debug("dev-screen capture for autodetect failed: %s", e)
        return None


def _probe(
    idx: int,
    n_frames: int = 4,
    inter_frame_s: float = 0.08,
    thumb_size: int = 128,
) -> Optional[dict]:
    """Open cv2 index ``idx``, capture ``n_frames`` frames, return a
    stats record or ``None`` if the index cannot be probed."""
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    try:
        frames = []
        for _ in range(n_frames):
            ok, f = cap.read()
            if not ok or f is None:
                return None
            frames.append(f.copy())
            time.sleep(inter_frame_s)
    finally:
        cap.release()
    if len(frames) < n_frames:
        return None
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    avg_std = float(stack.std(axis=0).mean())
    last_small = cv2.resize(
        frames[-1], (thumb_size, thumb_size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    return {"idx": idx, "avg_std": avg_std, "last_small": last_small}


# Process-wide cache: when several targets in targets.toml declare
# camera_index = "auto", we want one probe pass per process, not
# one per target.
_CACHED_INDEX: Optional[int] = None


def autodetect_camera_index(max_index: int = 7) -> int:
    """Pick the best cv2 camera index for a remote-control target.

    See module docstring for the selection rules. Returns ``0`` if no
    usable index is found, so callers can still construct a
    capture object that fails consistently downstream rather than
    crash in the autodetect itself.
    """
    global _CACHED_INDEX
    if _CACHED_INDEX is not None:
        return _CACHED_INDEX

    dev_thumb = _grab_dev_screen_thumb()
    candidates: list[dict] = []
    for idx in range(max_index + 1):
        p = _probe(idx)
        if p is None:
            continue
        if dev_thumb is not None:
            diff = float(np.abs(p["last_small"] - dev_thumb).mean())
            p["diff_vs_dev"] = diff
            is_dev_self = (
                p["avg_std"] < _STATIC_TEMPORAL_STD
                and diff < _DEV_SELF_DIFF_MAX
            )
            if is_dev_self:
                logger.info(
                    "autodetect: cv2 index %d matches dev mac own "
                    "screen (avg_std=%.3f, diff=%.2f) — skipping",
                    idx, p["avg_std"], diff,
                )
                continue
        candidates.append(p)

    if not candidates:
        logger.warning(
            "autodetect: no usable camera index found; defaulting to 0"
        )
        _CACHED_INDEX = 0
        return 0

    static = [c for c in candidates if c["avg_std"] < _STATIC_TEMPORAL_STD]
    chosen = static[0] if static else candidates[0]
    style = (
        "screen-share (static)"
        if chosen["avg_std"] < _STATIC_TEMPORAL_STD
        else "live (physical webcam)"
    )
    extra = (
        f", diff_vs_dev={chosen['diff_vs_dev']:.2f}"
        if "diff_vs_dev" in chosen else ""
    )
    logger.info(
        "autodetect: chose cv2 index %d (%s, avg_std=%.3f%s)",
        chosen["idx"], style, chosen["avg_std"], extra,
    )
    _CACHED_INDEX = int(chosen["idx"])
    return _CACHED_INDEX


def reset_cache() -> None:
    """Forget the cached autodetect result. Useful for tests and for
    re-probing after the operator plugs in a different camera."""
    global _CACHED_INDEX
    _CACHED_INDEX = None
