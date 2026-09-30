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
        return ObjectQA(status="retry", reason=obj.error or "missing asset or mask", retry_strategy="rerun_segmentation", failure_types=["BAD_MASK"])
    if obj.review_required:
        return ObjectQA(status="manual_review", reason="Protected low-confidence small object; inspect candidate lineage")
    if config.mock and obj.mock_retry_resolved:
        return ObjectQA(status="pass", reason="Mock retry simulation accepted this object; confidence is unchanged.")
    if max(obj.confidence, obj.classification_confidence or 0) < (obj.confidence_threshold if obj.confidence_threshold is not None else config.qa.min_confidence):
        return ObjectQA(status="retry", reason="confidence below configured minimum", retry_strategy="change_prompt", failure_types=["WRONG_CATEGORY"])
    if obj.metrics.get("occupancy", 0) < config.qa.min_occupancy:
        return ObjectQA(status="retry", reason="mask occupancy below configured minimum", retry_strategy="rerun_segmentation", failure_types=["MASK_TOO_SMALL"])
    outside_fraction = mask_outside_bbox(read_mask(obj.mask_path), obj.bbox.model_dump(), config.crop.alpha_threshold)
    if outside_fraction > config.qa.max_mask_outside_bbox:
        return ObjectQA(status="retry", reason="mask extends substantially outside its detector box", retry_strategy="rerun_segmentation", failure_types=["BACKGROUND_LEAK"])
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
                if obj.segmentation:
                    obj.metrics["segmentation_score"] = float(obj.segmentation.get("score", 0.0))
            qa = assess_object(obj, config)
            if config.p1.enabled and not config.mock:
                from qa.instance_qa import inspect_instance
                metrics, failures = inspect_instance(obj.model_dump(), state['source_path'], config)
                obj.metrics.update(metrics)
                qa.failure_types = list(dict.fromkeys([*qa.failure_types, *failures]))
                if qa.failure_types and qa.status == 'pass':
                    qa = ObjectQA(status='retry', reason=', '.join(qa.failure_types),
                                  retry_strategy='rerun_segmentation', failure_types=qa.failure_types)
                if 'WRONG_CATEGORY' in qa.failure_types:
                    qa.reason += '; requires classification evidence, segmentation cannot fix category'
                    if any(h.get('action') == 'classification_unavailable' for h in obj.retry_history):
                        qa.status = 'manual_review'
                elif 'CROSS_TILE_FRAGMENT' in qa.failure_types:
                    qa.retry_strategy = 'merge_neighbor_tiles'
                    if any(h.get('action') in {'fragment_detection_unavailable', 'fragment_detection_budget_exhausted'} for h in obj.retry_history):
                        qa.status = 'manual_review'
                if len(obj.retry_history) >= config.p1.max_segmentation_retry and qa.status == 'retry':
                    qa.status = 'manual_review'
                    qa.reason += '; P1 per-object retry budget exhausted'
            if qa.status == "retry":
                failed.append(obj.id)
                if exhausted:
                    qa.status = "manual_review"
                    qa.reason += "; retry budget exhausted"
            elif qa.status == "manual_review":
                failed.append(obj.id)
            obj.qa, obj.status = qa, qa.status
            objects.append(obj.model_dump(mode="json"))
        return {"objects": objects, "failed_objects": failed,
                'retryable_objects': [o['id'] for o in objects if o['status'] == 'retry'] if config.p1.enabled and not config.mock else failed}
    return qa_objects
