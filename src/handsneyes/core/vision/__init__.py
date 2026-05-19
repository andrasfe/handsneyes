"""Vision helpers: cursor + text finders, MLP loaders, image utils.

Phase A scope: enough to support tier-1 agents (verify/cursor/target)
and the tier-2 agents that don't need cursor homing. The full
visual-servo homer (commander/visual_servo_homer.py) is deliberately
deferred — it's a tier-3 click-engine concern and lands together with
the linux_gnome adapter and the homer's training-data persistence
machinery in Phase B.
"""

from handsneyes.core.vision.cursor_finder import (
    CursorHit,
    annotate_cursor,
    find_cursor_by_variance,
    find_cursor_hsv,
    find_cursor_hsv_motion,
    find_cursor_hsv_motion_directed,
    find_cursor_hsv_near,
)
from handsneyes.core.vision.imaging import (
    enhance_for_ocr,
    enhance_for_screen,
    numpy_to_base64_png,
    numpy_to_pil,
    pil_to_numpy,
    resize_for_mllm,
)
from handsneyes.core.vision.longjump import (
    LongJumpConfig,
    LongJumpModel,
    chunk_hid_for_bursts,
)
from handsneyes.core.vision.ocr_finder import (
    OCRHit,
    annotate_ocr_hit,
    find_text,
    have_ocr,
)
from handsneyes.core.vision.pointer_accel import (
    PointerAccelConfig,
    PointerAccelModel,
)
from handsneyes.core.vision.text_targeting import (
    showui_prompt_variants,
    target_keywords,
)

__all__ = [
    "CursorHit",
    "LongJumpConfig",
    "LongJumpModel",
    "OCRHit",
    "PointerAccelConfig",
    "PointerAccelModel",
    "annotate_cursor",
    "annotate_ocr_hit",
    "chunk_hid_for_bursts",
    "enhance_for_ocr",
    "enhance_for_screen",
    "find_cursor_by_variance",
    "find_cursor_hsv",
    "find_cursor_hsv_motion",
    "find_cursor_hsv_motion_directed",
    "find_cursor_hsv_near",
    "find_text",
    "have_ocr",
    "numpy_to_base64_png",
    "numpy_to_pil",
    "resize_for_mllm",
    "pil_to_numpy",
    "showui_prompt_variants",
    "target_keywords",
]
