"""Image normalization and gentle base enhancement."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_normalized(data: bytes) -> Image.Image:
    """Load image with EXIF orientation fix — critical for phone photos."""
    img = Image.open(__import__("io").BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def gentle_face_prep(image: Image.Image) -> Image.Image:
    """
    Light pre-processing before style LUTs.
    No CLAHE, no aggressive sharpen — avoids gritty skin and muddy tones.
    """
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # Bilateral filter: smooth skin while keeping edges
    smooth = cv2.bilateralFilter(bgr, d=5, sigmaColor=40, sigmaSpace=40)
    # Very subtle unsharp (5% detail blend)
    blur = cv2.GaussianBlur(smooth, (0, 0), 1.0)
    result = cv2.addWeighted(smooth, 1.05, blur, -0.05, 0)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def resize_for_preview(image: Image.Image, max_side: int = 480) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image.copy()
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
