"""RGBA alpha utilities."""
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cv.bbox import get_tight_bbox


def alpha_mask(image: Image.Image) -> NDArray[np.uint8]:
    """Extract a detached 8-bit alpha plane."""
    return np.array(image.convert("RGBA").getchannel("A"), dtype=np.uint8)


def alpha_bbox(image: Image.Image, threshold: int = 8) -> dict[str, int] | None:
    """Get the unpadded visible bounds of an RGBA asset."""
    return get_tight_bbox(alpha_mask(image), threshold, padding=0)
