"""DINOv2-driven click-target snapping.

After the homer's micro-correction loop drives the cursor to within
~1-5 px of the aim point, this module asks: "is there a UI element
near the aim that the user actually intended to click?" If yes, and
its centroid is within a small snap radius of the aim, return that
centroid so the homer can do one more nudge before clicking.

Why DINOv2:
  - Pre-trained self-supervised model (no labelled UI data needed)
  - Patch-level features cluster well by visual coherence — UI
    elements (buttons, links, fields) have internal feature
    consistency that's distinct from background
  - Small variant (21M params, ~85MB) is fast enough for online use

Algorithm (intentionally simple as a first prototype):
  1. Crop a ROI around the aim point (~224 px square, the
     resolution DINOv2 was trained at).
  2. Run DINOv2-small to get a 16×16 grid of 384-dim patch
     features.
  3. Identify the patch the aim point falls in; compute cosine
     similarity from that patch to every other patch.
  4. Threshold the similarity map → connected components of
     visually-coherent patches.
  5. The component containing the aim-point patch is "what the
     user is hovering over". Return its centroid in image-percent
     coordinates of the ORIGINAL frame.

This is a first-pass heuristic; the algorithm can absolutely be
improved (attention-rollout, multi-seed search, learned snap-
target classifier, etc.) but the lazy-loaded ViT + cosine
similarity is the smallest thing that demonstrates the idea.
"""
from __future__ import annotations
import logging
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "facebook/dinov2-small"
# DINOv2-small uses 14×14 patches of 14 px each = 196 patches for a
# 224 px image (officially 16×16 patches on a 224 image, but the
# small variant publishes patch_size=14 — confirm at load time).
_TARGET_RES = 224

_model = None
_processor = None
_device = None
_load_lock = threading.Lock()
_load_failed = False


def _try_load() -> bool:
    """Lazy-load DINOv2 + processor on first use. Idempotent.

    Returns True on success, False if any import / download / weight-
    load step fails. Heavy modules (torch, transformers) are imported
    INSIDE this function so the homer's import time stays cheap when
    dino-snap is disabled.
    """
    global _model, _processor, _device, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            logger.warning(
                "dino_snap: torch/transformers unavailable (%s) — "
                "install with `pip install torch transformers`. "
                "dino-snap will silently no-op.", e,
            )
            _load_failed = True
            return False
        try:
            _processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
            _model = AutoModel.from_pretrained(_MODEL_NAME)
            _model.eval()
            # Prefer MPS on Apple Silicon, then CUDA, then CPU.
            if torch.backends.mps.is_available():
                _device = torch.device("mps")
            elif torch.cuda.is_available():
                _device = torch.device("cuda")
            else:
                _device = torch.device("cpu")
            _model = _model.to(_device)
            logger.info(
                "dino_snap: loaded %s on %s", _MODEL_NAME, _device,
            )
            return True
        except Exception as e:
            logger.warning(
                "dino_snap: model load failed (%s). Will silently "
                "no-op for the rest of this session.", e,
            )
            _load_failed = True
            return False


def find_snap_target(
    frame_bgr: np.ndarray,
    aim_xy_pct: tuple[float, float],
    *,
    cursor_xy_pct: Optional[tuple[float, float]] = None,
    cursor_mask_radius_pct: float = 0.015,
    snap_radius_pct: float = 0.03,
    roi_size_px: int = _TARGET_RES,
    similarity_threshold: float = 0.55,
) -> Optional[tuple[float, float]]:
    """Snap the aim point to the nearest coherent UI element.

    Args:
        frame_bgr: Full webcam/capture-card frame, OpenCV BGR uint8.
        aim_xy_pct: (x, y) in image-percent — the click point the
            homer would commit if there's no snap to make.
        snap_radius_pct: Max distance in image-pct between aim and
            the snap target. Anything further is rejected (we don't
            want to teleport the click to a different UI element).
            Default 3% ≈ 60 px on 1920w ≈ ~half a button width.
        roi_size_px: Side of the square ROI to feed DINOv2. 224 is
            the resolution DINOv2 was trained at.
        similarity_threshold: Cosine-sim cutoff for "same UI
            element as the click point". Lower = more permissive
            (groups more patches together).

    Returns:
        (x_pct, y_pct) of the snap target, or None if:
            - dino model can't be loaded (no torch / weights / etc.)
            - no coherent component contains the aim-point patch
            - the resulting centroid is more than snap_radius_pct
              from the original aim
    """
    if not _try_load():
        return None

    import torch  # safe — _try_load succeeded

    h, w = frame_bgr.shape[:2]
    aim_px_x = int(aim_xy_pct[0] * w)
    aim_px_y = int(aim_xy_pct[1] * h)

    # Centred square ROI, clipped to frame bounds.
    half = roi_size_px // 2
    x0 = max(0, aim_px_x - half)
    y0 = max(0, aim_px_y - half)
    x1 = min(w, x0 + roi_size_px)
    y1 = min(h, y0 + roi_size_px)
    # Re-anchor x0/y0 to keep the ROI exactly roi_size_px square when
    # the aim was near the frame edge.
    x0 = max(0, x1 - roi_size_px)
    y0 = max(0, y1 - roi_size_px)
    roi_bgr = frame_bgr[y0:y1, x0:x1]
    if roi_bgr.shape[0] != roi_size_px or roi_bgr.shape[1] != roi_size_px:
        # Pad if the frame was smaller than 224 px in either axis —
        # unusual but possible during initial connection.
        roi_bgr = cv2.copyMakeBorder(
            roi_bgr,
            0, roi_size_px - roi_bgr.shape[0],
            0, roi_size_px - roi_bgr.shape[1],
            cv2.BORDER_REPLICATE,
        )

    # Mask out the cursor sprite if its position was provided. DINOv2
    # clusters by visual coherence — and the cursor itself is the
    # most visually-coherent thing inside the ROI, so without
    # masking it dominates the seed patch's connected component and
    # the snap centroid lands on the cursor's geometric centre
    # (down-right of the hotspot for an up-left-pointing arrow,
    # producing a systematic down-right snap bias on every click).
    # Painting a neutral-grey disk over the cursor before patch-
    # encoding kills its features so the snap reflects the UI under
    # the click rather than the cursor sprite obscuring it.
    if cursor_xy_pct is not None:
        cursor_in_roi_x = int(cursor_xy_pct[0] * w) - x0
        cursor_in_roi_y = int(cursor_xy_pct[1] * h) - y0
        if (
            0 <= cursor_in_roi_x < roi_size_px
            and 0 <= cursor_in_roi_y < roi_size_px
        ):
            mask_r = max(8, int(cursor_mask_radius_pct * w))
            cv2.circle(
                roi_bgr,
                (cursor_in_roi_x, cursor_in_roi_y),
                mask_r,
                # Neutral mid-grey — gives the patches under the
                # cursor low contrast against typical UI, so they
                # tend to cluster with background rather than with
                # any specific UI element.
                (127, 127, 127),
                thickness=-1,
            )

    # DINOv2 expects RGB.
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    inputs = _processor(images=roi_rgb, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
    # outputs.last_hidden_state shape: (1, 1+P*P, dim)
    # [0] = CLS token, [1:] = patch tokens in row-major order.
    patches = outputs.last_hidden_state[0, 1:, :]  # (P*P, dim)
    # Patch grid size derived from the actual output (DINOv2-small
    # uses patch_size=14 → 16×16 grid for a 224 input).
    grid = int(round(patches.shape[0] ** 0.5))
    if grid * grid != patches.shape[0]:
        logger.debug("dino_snap: unexpected patch count %d", patches.shape[0])
        return None

    # Find which patch the aim point falls in (within the ROI).
    aim_in_roi_x = aim_px_x - x0
    aim_in_roi_y = aim_px_y - y0
    patch_size_px = roi_size_px / grid
    aim_patch_col = min(grid - 1, max(0, int(aim_in_roi_x / patch_size_px)))
    aim_patch_row = min(grid - 1, max(0, int(aim_in_roi_y / patch_size_px)))
    aim_patch_idx = aim_patch_row * grid + aim_patch_col

    # Normalise features for cosine similarity.
    patches_n = patches / (
        patches.norm(dim=-1, keepdim=True) + 1e-6
    )
    seed = patches_n[aim_patch_idx]
    sims = (patches_n @ seed).cpu().numpy().reshape(grid, grid)

    # Threshold + connected components in the patch grid.
    mask = (sims >= similarity_threshold).astype(np.uint8)
    if mask.sum() == 0:
        return None
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    aim_label = labels[aim_patch_row, aim_patch_col]
    if aim_label == 0:
        # Aim patch wasn't in any high-similarity component — the
        # seed patch is an outlier. Nothing to snap to.
        return None

    rows, cols = np.where(labels == aim_label)
    if len(rows) == 0:
        return None

    # Background rejection. If the component covers more than half
    # the ROI it's almost certainly the page background (white
    # space, panel body, modal backdrop), NOT a discrete clickable
    # UI element. Its centroid is wherever the background extends
    # — which produces a systematic snap bias toward the page's
    # centre of mass rather than any actual click target. Reject
    # those and let the homer commit at its geometric aim.
    total_patches = grid * grid
    if len(rows) > 0.45 * total_patches:
        return None

    # Centroid of the component in patch-grid coords, then back to
    # full-frame pixels.
    cy_patch = float(rows.mean())
    cx_patch = float(cols.mean())
    cx_in_roi_px = (cx_patch + 0.5) * patch_size_px
    cy_in_roi_px = (cy_patch + 0.5) * patch_size_px
    snap_px_x = x0 + cx_in_roi_px
    snap_px_y = y0 + cy_in_roi_px
    snap_pct = (snap_px_x / w, snap_px_y / h)

    # Reject if the snap target moved further than snap_radius_pct
    # — we don't want to jump to a different UI element entirely.
    dist = float(
        ((snap_pct[0] - aim_xy_pct[0]) ** 2
         + (snap_pct[1] - aim_xy_pct[1]) ** 2) ** 0.5
    )
    if dist > snap_radius_pct:
        return None

    return snap_pct
