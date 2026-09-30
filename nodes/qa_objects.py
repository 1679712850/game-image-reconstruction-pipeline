"""V1 object QA is deterministic; no LLM is invoked."""
from collections.abc import Callable

from agent.state import SceneState
from app.config import PipelineConfig
from cv.mask import read_mask
from cv.metrics import bbox_area, occupancy, touches_edge, mask_outside_bbox
from schemas.object import SceneObject
from schemas.qa import ObjectQA


def assess_object(obj: SceneObject, config: PipelineConfig) -> ObjectQA:
    """Explain each failure and select a bounded repair strategy."""
    if obj.error or not obj.asset_path or not obj.mask_path:
        return ObjectQA(status="retry", reason=obj.error or "missing asset or mask", retry_strategy="rerun_segmentation")
    if obj.review_required:
        return ObjectQA(status="manual_review", reason="Protected low-confidence small object; inspect candidate lineage")
    if config.mock and obj.mock_retry_resolved:
        return ObjectQA(status="pass", reason="Mock retry simulation accepted this object; confidence is unchanged.")
    if obj.confidence < (obj.confidence_threshold if obj.confidence_threshold is not None else config.qa.min_confidence):
        return ObjectQA(status="retry", reason="confidence below configured minimum", retry_strategy="change_prompt")
    if obj.metrics.get("occupancy", 0) < config.qa.min_occupancy:
        return ObjectQA(status="retry", reason="mask occupancy below configured minimum", retry_strategy="rerun_segmentation")
    outside_fraction = mask_outside_bbox(read_mask(obj.mask_path), obj.bbox.model_dump(), config.crop.alpha_threshold)
    if outside_fraction > config.qa.max_mask_outside_bbox:
        return ObjectQA(status="retry", reason="mask extends substantially outside its detector box", retry_strategy="rerun_segmentation")
    return ObjectQA(status="pass", reason="confidence and mask occupancy passed")


def make_qa_objects(config: PipelineConfig) -> Callable[[SceneState], dict]:
    """Bind thresholds and mock policy once."""
    def qa_objects(state: SceneState) -> dict:
        """Return QA reasons, metrics, and unresolved IDs for the router."""
        objects, failed = [], []
        exhausted = state.get("retry_count", 0) >= state.get("max_retry", config.max_retry)
        for record in state.get("objects", []):
            obj = SceneObject.model_validate(record)
            if obj.mask_path:
                mask = read_mask(obj.mask_path)
                box = (obj.crop_bbox or obj.bbox).model_dump()
                obj.metrics = {
                    "occupancy": occupancy(mask, box, config.crop.alpha_threshold),
                    "bbox_area": bbox_area(box),
                    "touches_edge": touches_edge(mask, config.crop.alpha_threshold),
                }
            qa = assess_object(obj, config)
            if qa.status == "retry":
                failed.append(obj.id)
                if exhausted:
                    qa.status = "manual_review"
                    qa.reason += "; retry budget exhausted"
            elif qa.status == "manual_review":
                failed.append(obj.id)
            obj.qa, obj.status = qa, qa.status
            objects.append(obj.model_dump(mode="json"))
        return {"objects": objects, "failed_objects": failed}
    return qa_objects
