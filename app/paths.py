"""Filesystem conventions shared by nodes and exporters."""
from pathlib import Path

from PIL import Image


def prepare_output(output_dir: str | Path) -> Path:
    """Create the per-scene artifact directories."""
    root = Path(output_dir).expanduser().resolve()
    for name in ("assets", "assets_hd", "masks", "debug"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def read_rgba(path: str | Path) -> Image.Image:
    """Read detached RGBA pixels and promptly close the file handle."""
    with Image.open(Path(path)) as image:
        return image.convert("RGBA")


def relative_asset(path: str | None, root: Path) -> str | None:
    """Serialize portable POSIX paths, restricted to the output directory."""
    if path is None:
        return None
    return Path(path).resolve().relative_to(root.resolve()).as_posix()
