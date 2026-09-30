"""Distinguish artificial tile boundaries from source-image boundaries."""
from schemas.detection_candidate import DetectionCandidate


def truncated_edges(bbox, window, image_size, threshold=12) -> list[str]:
    x0, y0, x1, y1 = bbox
    wx0, wy0, wx1, wy1 = window
    width, height = image_size
    return [name for name, hit in (
        ("left", wx0 > 0 and x0-wx0 <= threshold),
        ("top", wy0 > 0 and y0-wy0 <= threshold),
        ("right", wx1 < width and wx1-x1 <= threshold),
        ("bottom", wy1 < height and wy1-y1 <= threshold),
    ) if hit]


def expanded_region(candidate: DetectionCandidate, size: tuple[int, int], padding: int):
    x0, y0, x1, y1 = candidate.bbox
    return max(0, x0-padding), max(0, y0-padding), min(size[0], x1+padding), min(size[1], y1+padding)
