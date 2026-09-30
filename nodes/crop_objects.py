"""Extract real tight RGBA object assets."""
from collections.abc import Callable
from pathlib import Path

from agent.state import SceneState
from app.config import CropConfig
from app.objects import crop_records


def make_crop_objects(config: CropConfig) -> Callable[[SceneState], dict]:
    """Bind crop options outside serialized state."""
    def crop_objects(state: SceneState) -> dict:
        """Create compact assets and preserve global crop origins."""
        version = f"_r{state['detection_round']:02d}" if state.get("detection_round", 0) > 1 else ""
        return {"objects": crop_records(state["source_path"], state["objects"], Path(state["output_dir"]), config, version=version)}
    return crop_objects
