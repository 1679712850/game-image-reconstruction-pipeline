"""Future Godot exporter; no game engine dependency in V1."""
from pathlib import Path

from schemas.scene import SceneManifest


def export_godot(scene: SceneManifest, output_dir: Path) -> Path:
    """TODO: export logical transforms, textures and Y-sortable scene nodes."""
    raise NotImplementedError("Godot export is reserved for a later implementation")
