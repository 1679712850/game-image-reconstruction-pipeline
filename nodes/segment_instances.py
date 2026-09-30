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
        version = f"_r{state['detection_round']:02d}" if state.get("detection_round", 0) > 1 else ""
        objects = segment_records(state.get("working_path", state["source_path"]), state["detections"], service, Path(state["output_dir"]),
                                  coverage_path=state.get("coverage_mask_path"), version=version)
        return {"objects": objects}
    return segment_instances
