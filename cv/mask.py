"""Mask I/O and conservative, optional morphological refinement."""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def as_mask(mask: NDArray | Image.Image) -> NDArray[np.uint8]:
    """Normalize a 2D uint8/bool mask; reject ambiguous RGB masks."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    if not np.isfinite(array).all():
        raise ValueError("Mask contains non-finite values")
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    return np.clip(array, 0, 255).astype(np.uint8)


def read_mask(path: str | Path) -> NDArray[np.uint8]:
    """Read an 8-bit grayscale PNG with a detached pixel buffer."""
    with Image.open(Path(path)) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


def save_mask(mask: NDArray, path: str | Path) -> str:
    """Persist mask values as grayscale, not as a separate alpha channel."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(as_mask(mask)).save(target, "PNG")
    return str(target.resolve())


def rectangle_mask(width: int, height: int, bbox: dict[str, int]) -> NDArray[np.uint8]:
    """Create a clipped rectangular mask in original scene coordinates."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = (bbox[key] for key in ("x", "y", "w", "h"))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def refine_mask(mask: NDArray, threshold: int = 8, close_kernel: int = 0) -> NDArray[np.uint8]:
    """Remove near-zero alpha; optionally close holes with OpenCV."""
    result = as_mask(mask).copy()
    result[result <= threshold] = 0
    if close_kernel:
        if close_kernel < 1 or close_kernel % 2 == 0:
            raise ValueError("close_kernel must be a positive odd integer")
        try:
            import cv2
            kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        except ImportError as error:
            raise RuntimeError("OpenCV is required when close_kernel is enabled") from error
    return result
