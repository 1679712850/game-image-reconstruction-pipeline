"""Reconstruction and a full-canvas similarity diagnostic."""
from collections.abc import Callable
from pathlib import Path

from agent.state import SceneState
from app.paths import read_rgba
from cv.metrics import reconstruction_similarity
from cv.reconstruct import reconstruct_scene as rebuild


def make_reconstruct_scene(enabled: bool) -> Callable[[SceneState], dict]:
    """Allow reconstruction to be disabled without altering graph structure."""
    def reconstruct_scene(state: SceneState) -> dict:
        """Rebuild from logical-size assets; missing scene regions stay transparent."""
        if not enabled:
            return {"reconstruction_path": "", "reconstruction_score": 0.0}
        path = rebuild(
            state["width"], state["height"], state.get("objects", []),
            Path(state["output_dir"]) / "reconstruction.png",
        )
        score = reconstruction_similarity(read_rgba(state["source_path"]), read_rgba(path))
        return {"reconstruction_path": path, "reconstruction_score": score}
    return reconstruct_scene
