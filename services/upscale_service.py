"""Replaceable Real-ESRGAN adapter with a genuine PIL Lanczos mock."""
from pathlib import Path

from PIL import Image

from app.paths import read_rgba


def choose_scale(width: int, height: int) -> int:
    """Use x4 below 128, x3 through 256, x2 through 512, otherwise x1."""
    edge = max(width, height)
    if edge < 128:
        return 4
    if edge <= 256:
        return 3
    if edge <= 512:
        return 2
    return 1


class UpscaleService:
    """Scale RGBA pixels without changing scene-space metadata."""

    def __init__(self, mock: bool = True):
        self.mock = mock

    def upscale(self, image_path: str, scale: float) -> str:
        """Write assets_hd/<id>@<scale>x.png and return its absolute path."""
        if not self.mock:
            raise NotImplementedError("TODO: connect Real-ESRGAN in UpscaleService.upscale")
        if scale <= 0:
            raise ValueError("Texture scale must be positive")
        source = Path(image_path)
        image = read_rgba(source)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        result = image.resize(size, Image.Resampling.LANCZOS)
        root = source.parent.parent if source.parent.name == "assets" else source.parent
        target = root / "assets_hd" / f"{source.stem}@{scale:g}x.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, "PNG")
        return str(target.resolve())
