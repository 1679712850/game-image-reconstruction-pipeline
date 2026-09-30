"""Save original-resolution masks produced by the injected SAM adapter."""
from collections.abc import Callable
from pathlib import Path

from agent.state import SceneState
from app.objects import segment_records
from services.sam_service import SAMService


def make_segment_instances(service: SAMService) -> Callable[[SceneState], dict]:
    """Bind a segmenter while keeping its arrays out of checkpoint state."""
    def segment_instances(state: SceneState) -> dict:
        """Return persisted segmentation records."""
        objects = segment_records(state["source_path"], state["detections"], service, Path(state["output_dir"]))
        return {"objects": objects}
    return segment_instances
