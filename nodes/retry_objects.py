"""A bounded, targeted repair pass; successful objects are untouched."""
from collections.abc import Callable
from pathlib import Path

from agent.state import SceneState
from app.config import PipelineConfig
from app.objects import crop_records, refine_records, segment_records
from services.sam_service import SAMService


def make_retry_objects(config: PipelineConfig, service: SAMService) -> Callable[[SceneState], dict]:
    """Bind retry dependencies; a future diagnosis agent can choose repairs."""
    def retry_objects(state: SceneState) -> dict:
        """Repair failed masks/crops and increment the retry counter exactly once."""
        count = state.get("retry_count", 0)
        if count >= state.get("max_retry", config.max_retry):
            return {}  # Defensive guard in addition to the router.
        failed = set(state.get("failed_objects", []))
        selected = [obj for obj in state["objects"] if obj["id"] in failed]
        root = Path(state["output_dir"])
        repaired = segment_records(state["source_path"], selected, service, root)
        repaired = refine_records(repaired, config.crop.alpha_threshold)
        repaired = crop_records(state["source_path"], repaired, root, config.crop)
        by_id = {}
        for obj in repaired:
            obj["mock_retry_resolved"] = config.mock and not obj["error"]
            if obj["mock_retry_resolved"]:
                obj["status"] = "pass"
            by_id[obj["id"]] = obj
        return {
            "retry_count": count + 1,
            "objects": [by_id.get(obj["id"], dict(obj)) for obj in state["objects"]],
        }
    return retry_objects
