"""Tight bounding boxes in original scene pixel coordinates."""
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cv.mask import as_mask


def get_tight_bbox(
    mask: NDArray | Image.Image, threshold: int = 8, padding: int = 16,
) -> dict[str, int] | None:
    """Return a clipped xywh box around mask > threshold, or None if empty."""
    if padding < 0 or not 0 <= threshold <= 254:
        raise ValueError("padding must be nonnegative and threshold in [0, 254]")
    array = as_mask(mask)
    height, width = array.shape
    ys, xs = np.nonzero(array > threshold)
    if xs.size == 0:
        return None
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
