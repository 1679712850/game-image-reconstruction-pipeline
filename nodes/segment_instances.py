"""Save original-resolution masks produced by the injected SAM adapter."""
from collections.abc import Callable
from pathlib import Path
import logging

from agent.state import SceneState
from app.objects import segment_records
from services.sam_service import SAMService
from app.detection_config import DetectionConfig
from fusion.mask_fusion import fuse_segmented_records
from cv.mask import read_mask


def make_segment_instances(service: SAMService, detection: DetectionConfig | None = None, p1=None) -> Callable[[SceneState], dict]:
    """Bind a segmenter while keeping its arrays out of checkpoint state."""
    def segment_instances(state: SceneState) -> dict:
        """Return persisted segmentation records."""
        version = f"_r{state['detection_round']:02d}" if state.get("detection_round", 0) > 1 else ""
        logging.getLogger(__name__).info("[Segmentation] candidates=%d", len(state["detections"]))
        records = state["detections"]
        if p1 is not None and p1.enabled:
            from scene.element_classifier import SceneElementClassifier
            records = SceneElementClassifier().classify_all(records, (state["width"], state["height"]))
        source = state["source_path"] if p1 and p1.enabled else state.get("working_path", state["source_path"])
        objects = segment_records(source, records, service, Path(state["output_dir"]),
                                  coverage_path=state.get("coverage_mask_path") if not (p1 and p1.enabled) else None,
                                  version=version, p1=p1)
        if detection is not None:
            objects, rejected = fuse_segmented_records(objects, detection.dedup)
            for obj in objects:
                reason = "segmentation_failed" if obj.get("error") else (
                    "mask_too_small" if obj.get("mask_path") and not read_mask(obj["mask_path"]).any() else None)
                if reason:
                    rejected.append({"id": obj["id"], "category": obj["category"], "confidence": obj["confidence"],
                                     "source_candidates": obj.get("source_candidates", []), "reason": reason,
                                     "stage": "segmentation", "round": state.get("detection_round", 1), "error": obj.get("error")})
            runs = list(state.get("detection_runs", []))
            if runs:
                runs[-1] = {**runs[-1], "filtered": [*runs[-1].get("filtered", []), *rejected]}
            return {"objects": objects, "detection_runs": runs}
        return {"objects": objects}
    return segment_instances
