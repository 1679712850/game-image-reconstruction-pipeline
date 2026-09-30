"""Extract compact RGBA assets without changing their original RGB."""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.paths import read_rgba
from cv.bbox import get_tight_bbox
from cv.mask import as_mask


def crop_rgba_by_mask(
    source_image: Image.Image | str | Path,
    mask: NDArray | Image.Image,
    bbox: dict[str, int] | None = None,
    output_path: str | Path | None = None,
    *, threshold: int = 8, padding: int = 16,
) -> Image.Image | None:
    """Return a tight RGBA crop; an empty mask returns None without writing."""
    source = read_rgba(source_image) if isinstance(source_image, (str, Path)) else source_image.convert("RGBA")
    array = as_mask(mask)
    if array.shape != (source.height, source.width):
        raise ValueError("Mask dimensions must match the source image")
    if not np.any(array > threshold):
        return None
    box = bbox or get_tight_bbox(array, threshold, padding)
    x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > source.width or y + h > source.height:
        raise ValueError("Crop bbox is outside the source image")
    crop = source.crop((x, y, x + w, y + h))
    crop.putalpha(Image.fromarray(array[y:y + h, x:x + w]))
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target, "PNG")
    return crop
