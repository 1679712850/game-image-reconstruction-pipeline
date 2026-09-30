"""Logical-pixel pivots for isometric Y sorting."""
import numpy as np
from numpy.typing import NDArray

from cv.mask import as_mask


def bottom_center(width: int, height: int) -> dict[str, float]:
    """Use a simple bottom-center fallback in the original crop."""
    return {"x": width / 2, "y": height * 0.95}


def ground_pivot(alpha: NDArray, threshold: int = 8) -> dict[str, float]:
    """Use the mean x of bottommost visible pixels, in pixel-center units."""
    array = as_mask(alpha)
    ys, xs = np.nonzero(array > threshold)
    if not xs.size:
        return bottom_center(array.shape[1], array.shape[0])
    bottom = int(ys.max())
    return {"x": float(xs[ys == bottom].mean() + 0.5), "y": bottom + 0.5}
