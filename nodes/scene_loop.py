"""Commit round assets, measure coverage and ask the bounded scene reviewer."""
from collections.abc import Callable
import json
from pathlib import Path

from agent.state import SceneState
from app.config import PipelineConfig
from cv.coverage import write_remaining
from schemas.scene_qa import SceneReviewDecision
from services.scene_review_service import SceneReviewService


def _candidate_quality(obj: dict) -> tuple[bool, bool, float]:
    """Prefer accepted masks, then usable crops, then detection confidence."""
    return obj["status"] == "pass", bool(obj.get("asset_path") and not obj.get("error")), obj["confidence"]


def make_update_remaining(config: PipelineConfig) -> Callable[[SceneState], dict]:
    """Keep all candidate assets, but remove only successful masks from detection."""
    def update_remaining(state: SceneState) -> dict:
        previous = {obj["id"]: obj for obj in state.get("archived_objects", [])}
        merged = dict(previous)
        for obj in state.get("objects", []):
            old = merged.get(obj["id"])
            # Keep the better candidate; an empty re-segmentation must not lose a valid crop.
            if old is None or (old["status"] != "pass" and _candidate_quality(obj) >= _candidate_quality(old)):
                merged[obj["id"]] = obj
        objects = list(merged.values())
        round_index = state["detection_round"]
        directory = Path(state["output_dir"]) / "debug" / f"round_{round_index:02d}"
        coverage = write_remaining(state["source_path"], objects, directory, config.crop.alpha_threshold)
        gain = coverage["accepted_coverage"] - state.get("scene_coverage", 0.0)
        new_accepted = sum(obj["status"] == "pass" and previous.get(obj["id"], {}).get("status") != "pass" for obj in objects)
        stalled = new_accepted == 0 or gain < config.scene_loop.min_coverage_gain
        metrics = {**coverage, "round": round_index, "new_accepted": new_accepted,
                   "coverage_gain": gain, "object_count": len(objects),
                   "manual_review_count": sum(obj["status"] != "pass" for obj in objects),
                   "target_coverage": config.scene_loop.target_coverage,
                   "definition": "Union of mask pixels over nontransparent source pixels; not semantic recall"}
        return {"objects": objects, "archived_objects": objects,
                "detections": state.get("all_detections", state.get("detections", [])),
                "failed_objects": [obj["id"] for obj in objects if obj["status"] != "pass"],
                "working_path": coverage["working_path"], "coverage_mask_path": coverage["coverage_mask_path"],
                "scene_coverage": coverage["accepted_coverage"], "scene_coverage_gain": gain,
                "scene_no_progress": state.get("scene_no_progress", 0) + 1 if stalled else 0,
                "scene_qa": metrics,
                "total_retry_count": state.get("total_retry_count", 0) + state.get("retry_count", 0)}
    return update_remaining


def make_qa_scene(config: PipelineConfig, reviewer: SceneReviewService) -> Callable[[SceneState], dict]:
    """Agent proposes; hard iteration, stall and asset limits have final authority."""
    def qa_scene(state: SceneState) -> dict:
        limits = config.scene_loop
        metrics = {k: v for k, v in state["scene_qa"].items() if not k.endswith("path")}
        failure = None
        try:
            decision = reviewer.review(state["source_path"], state["working_path"], metrics)
        except Exception as error:
            # No silent LLM -> rules fallback. Export existing work for manual review.
            failure = type(error).__name__
            decision = SceneReviewDecision(continue_detection=False, status="manual_review",
                                           reason=f"Scene reviewer failed ({failure}); inspect provider connectivity/schema")
        hard_stop = ""
        if state["detection_round"] >= limits.max_rounds:
            hard_stop = "max_rounds"
        elif len(state["objects"]) >= limits.max_objects:
            hard_stop = "max_objects"
        elif state.get("scene_no_progress", 0) >= limits.no_progress_patience:
            hard_stop = "no_progress"
        more = decision.continue_detection and not hard_stop and failure is None
        if config.p1.enabled and not config.mock and more:
            # P1 performs problem-region detection after scene QA instead of another whole-image pass.
            more = False
            hard_stop = 'targeted_p1_review'
        report = {**metrics, "backend": reviewer.backend, "decision": decision.model_dump(),
                  "continue_detection": more, "stop_reason": hard_stop or ("" if more else "reviewer_stop"),
                  "reviewer_error": failure,
                  "status": "needs_detection" if more else (
                      "manual_review" if hard_stop or failure or state["failed_objects"] else decision.status)}
        directory = Path(state["output_dir"]) / "debug" / f"round_{state['detection_round']:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scene_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"scene_continue": bool(more), "scene_stop_reason": report["stop_reason"], "scene_qa": report,
                "retry_count": state["retry_count"] if more else state.get("total_retry_count", state["retry_count"]),
                "scene_history": [*state.get("scene_history", []), report],
                "scene_next_categories": decision.suggested_categories}
    return qa_scene
