"""PNG export primitives, shared by future asset exporters."""
from pathlib import Path

from PIL import Image


def export_png(image: Image.Image, path: str | Path) -> str:
    """Write a real RGBA PNG and return its absolute path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGBA").save(target, "PNG")
    return str(target.resolve())
