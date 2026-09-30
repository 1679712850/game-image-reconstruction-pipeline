"""Detection node with stable IDs, independent from model initialization."""
from collections import Counter
from collections.abc import Callable
import re

from agent.state import SceneState
from schemas.object import SceneObject
from services.grounding_service import GroundingService


def make_detect_instances(service: GroundingService, exercise_retry: bool = False) -> Callable[[SceneState], dict]:
    """Bind a detector and optionally inject one demonstrable mock failure."""
    def detect_instances(state: SceneState) -> dict:
        """Return schema-validated detections in source-image coordinates."""
        categories = list(dict.fromkeys(
            category for layer in state["layer_plan"] for category in layer["categories"]
        ))
        raw = service.detect(state["source_path"], categories)
        counts, detections = Counter(), []
        for index, item in enumerate(raw):
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", item["category"]).strip("_") or "object"
            counts[slug] += 1
            obj = SceneObject(
                id=f"{slug}_{counts[slug]:03d}", category=item["category"],
                confidence=0.1 if exercise_retry and index == 0 else item["confidence"],
                bbox=item["bbox"],
            )
            if obj.bbox.x + obj.bbox.w > state["width"] or obj.bbox.y + obj.bbox.h > state["height"]:
                raise ValueError(f"Detection bbox lies outside source: {obj.id}")
            detections.append(obj.model_dump(mode="json"))
        return {"detections": detections}
    return detect_instances
