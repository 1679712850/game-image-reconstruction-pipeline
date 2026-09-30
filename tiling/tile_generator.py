"""Deterministic sliding-window tile generation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """A tile in original-image coordinates."""

    tile_id: str
    x: int
    y: int
    width: int
    height: int
    scale: int | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def as_dict(self) -> dict:
        return {"tile_id": self.tile_id, "x": self.x, "y": self.y,
                "width": self.width, "height": self.height,
                **({"scale": self.scale} if self.scale is not None else {})}


def _starts(length: int, size: int, overlap: float) -> list[int]:
    last = max(0, length - size)
    stride = max(1, round(size * (1 - overlap)))
    return sorted(set([*range(0, last + 1, stride), last]))


def generate_tiles(width: int, height: int, tile_size: int = 1024,
                   overlap: float = 0.25, *, scale: int | None = None) -> list[Tile]:
    """Return overlapping tiles covering every source pixel."""
    if min(width, height, tile_size) <= 0 or not 0 < overlap < 1:
        raise ValueError("Positive dimensions and overlapping tiles (0 < overlap < 1) required")
    xs, ys = _starts(width, tile_size, overlap), _starts(height, tile_size, overlap)
    return [Tile(f"tile_{row:02d}_{col:02d}", x, y,
                 min(tile_size, width - x), min(tile_size, height - y), scale)
            for row, y in enumerate(ys) for col, x in enumerate(xs)]
