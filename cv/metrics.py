"""Deterministic mask QA and full-canvas reconstruction metrics."""
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cv.mask import as_mask


def bbox_area(bbox: dict[str, int]) -> int:
    """Calculate the pixel area of a nonnegative xywh rectangle."""
    return max(0, bbox["w"]) * max(0, bbox["h"])


def occupancy(mask: NDArray, bbox: dict[str, int] | None = None, threshold: int = 8) -> float:
    """Count valid pixels divided by bbox area (or mask area if omitted)."""
    array = as_mask(mask)
    height, width = array.shape
    box = bbox or {"x": 0, "y": 0, "w": width, "h": height}
    area = bbox_area(box)
    if area == 0:
        return 0.0
    x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(array[y0:y1, x0:x1] > threshold) / area)


def touches_edge(mask: NDArray, threshold: int = 8) -> bool:
    """Report whether valid pixels touch any border of the supplied mask."""
    array = as_mask(mask)
    if array.size == 0:
        return False
    return bool(any(np.any(edge > threshold) for edge in (
        array[0], array[-1], array[:, 0], array[:, -1],
    )))


def mask_outside_bbox(mask: NDArray, bbox: dict[str, int], threshold: int = 8) -> float:
    """Fraction of foreground pixels outside a detector box, independent of occupancy."""
    active = as_mask(mask) > threshold
    count = int(np.count_nonzero(active))
    if not count:
        return 0.0
    x, y, w, h = (bbox[key] for key in ("x", "y", "w", "h"))
    inside = int(np.count_nonzero(active[max(0, y):max(0, y+h), max(0, x):max(0, x+w)]))
    return 1.0 - inside / count


def reconstruction_similarity(source: Image.Image, preview: Image.Image) -> float:
    """Return 1 - normalized full-canvas RGBA MAE, including missing regions."""
    if source.size != preview.size:
        raise ValueError("Reconstruction size does not match the source")
    expected = np.asarray(source.convert("RGBA"), dtype=np.float32)
    actual = np.asarray(preview.convert("RGBA"), dtype=np.float32)
    return float(1.0 - np.abs(expected - actual).mean() / 255.0)
