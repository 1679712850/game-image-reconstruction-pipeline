"""Explicit conversion between local, normalized and global coordinates."""
from collections.abc import Sequence
import math


def to_pixel_bbox(bbox: Sequence[float], width: int, height: int,
                  normalized: bool = False) -> tuple[int, int, int, int]:
    """Convert xyxy to clipped integer pixel coordinates."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values")
    if not all(math.isfinite(v) for v in bbox):
        raise ValueError("bbox must be finite")
    if normalized and not all(0 <= v <= 1 for v in bbox):
        raise ValueError("Normalized coordinates must lie in [0, 1]")
    if normalized:
        values = (bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height)
    else:
        values = tuple(bbox)
    x0, y0, x1, y1 = values
    x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox has no positive area")
    return x0, y0, x1, y1


def restore_bbox(local_bbox: Sequence[float], tile_x: int, tile_y: int,
                 tile_width: int, tile_height: int, *, normalized: bool = False) -> dict[str, int]:
    """Restore a local xyxy box into absolute source-image xywh coordinates."""
    x0, y0, x1, y1 = to_pixel_bbox(local_bbox, tile_width, tile_height, normalized)
    return {"x": tile_x + x0, "y": tile_y + y0, "w": x1 - x0, "h": y1 - y0}
