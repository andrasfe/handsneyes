"""Image processing utilities.

Shared image encoding, conversion, and preprocessing functions used
by the capture and interpreter modules.
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def numpy_to_base64_png(image: np.ndarray) -> str:
    """Convert a numpy image array (BGR, OpenCV format) to base64 PNG."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a numpy image array (BGR) to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to a numpy array (BGR)."""
    rgb_array = np.array(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhance a camera-captured terminal image for better MLLM OCR.

    Produces high-contrast black text on white background regardless
    of input polarity (works for both white-on-black and black-on-white
    displays).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold to get binary text
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure black-on-white: if majority of pixels are dark, invert
    white_ratio = np.mean(binary) / 255.0
    if white_ratio < 0.5:
        binary = cv2.bitwise_not(binary)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_for_screen(image: np.ndarray) -> np.ndarray:
    """Light enhancement for webcam photos of screens.

    Only applies mild CLAHE to even out lighting differences.
    No brightness/contrast boost — raw webcam images are usually fine.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_for_mllm(
    image: np.ndarray,
    max_dimension: int = 1568,
    min_dimension: int = 1024,
) -> np.ndarray:
    """Resize an image for optimal MLLM interpretation.

    Preserves aspect ratio. Downscales large images and upscales
    small images so text is readable by the vision model.
    """
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest > max_dimension:
        scale = max_dimension / largest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
    elif largest < min_dimension:
        scale = min_dimension / largest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )

    return image
