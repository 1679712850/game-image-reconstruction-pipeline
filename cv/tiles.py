"""Deterministic overlapping windows in original image coordinates."""


def tile_windows(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return xyxy windows covering every pixel, with aligned final edges."""
    if min(width, height, size) <= 0 or not 0 <= overlap < 1:
        raise ValueError("Positive dimensions and overlap in [0, 1) required")

    def starts(length: int) -> list[int]:
        last = max(0, length - size)
        values = list(range(0, last + 1, max(1, round(size * (1 - overlap)))))
        return sorted(set([*values, last]))

    return [(x, y, min(x + size, width), min(y + size, height))
            for y in starts(height) for x in starts(width)]
