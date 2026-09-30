"""Load and validate the scene before any model service runs."""
from pathlib import Path

from agent.state import SceneState
from app.paths import prepare_output, read_rgba


def load_image(state: SceneState) -> dict:
    """Return source dimensions and canonical paths without mutating state."""
    path = Path(state["source_path"]).expanduser().resolve()
    source = read_rgba(path)
    if source.width < 1 or source.height < 1:
        raise ValueError("Input image must have positive dimensions")
    root = prepare_output(state["output_dir"])
    return {"source_path": str(path), "output_dir": str(root), "width": source.width, "height": source.height}
