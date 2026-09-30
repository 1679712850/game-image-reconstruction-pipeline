"""Future PSD adapter; deliberately excluded from the V1 graph."""
from pathlib import Path

from schemas.scene import SceneManifest


def export_psd(scene: SceneManifest, output_path: Path) -> Path:
    """TODO: map semantic layers, object order and offsets to PSD layers."""
    raise NotImplementedError("PSD export is reserved for a later implementation")
