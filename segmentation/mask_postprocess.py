"""Keep disconnected foliage and one-pixel rods; close only solid categories."""
import cv2
import numpy as np
from cv.mask import as_mask

SOLID = {"building", "house", "temple", "rock", "large_rock", "medium_rock", "small_rock", "statue"}


def mask_postprocess_by_category(mask, category, threshold=8):
    result = as_mask(mask)
    result = (result > threshold).astype(np.uint8) * 255
    if category in SOLID:
        # Only fill single-pixel defects, without dilating an outer boundary.
        closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        support = cv2.dilate(result, np.ones((3, 3), np.uint8)) > 0
        result = np.where(support, closed, result).astype(np.uint8)
    return result
