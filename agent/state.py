"""JSON-serializable state; images and models stay outside checkpoints."""
from typing import TypedDict


class SceneState(TypedDict, total=False):
    """Small, readable state shared by all workflow nodes."""

    source_path: str
    output_dir: str
    width: int
    height: int
    scene_analysis: dict
    layer_plan: list[dict]
    detections: list[dict]
    objects: list[dict]
    failed_objects: list[str]
    retryable_objects: list[str]
    retry_count: int
    max_retry: int
    reconstruction_path: str
    reconstruction_score: float | None
    scene_json: str
    exported_assets: list[str]
    decomposed_layers: list[dict]
    edit_requests: list[dict]
    object_edits: list[dict]
    working_path: str
    detection_round: int
    scene_coverage: float
    scene_coverage_gain: float
    scene_no_progress: int
    scene_qa: dict
    archived_objects: list[dict]
    scene_history: list[dict]
    coverage_mask_path: str
    scene_next_categories: list[str]
    scene_continue: bool
    scene_stop_reason: str
    detection_diagnostics: dict
    detection_runs: list[dict]
    detection_budget: dict
    total_retry_count: int
    all_detections: list[dict]
    ownership: dict
    terrain_layers: list[dict]
    retry_history: list[dict]
    missed_object_candidates: list[dict]
    detection_coverage_review: dict
    p1_thresholds: dict
    p1_summary: dict
    completion_metrics: dict
    pipeline_status: str
    mask_metrics: dict

    candidate_registry: dict
    performance_report_path: str
    timeline_path: str
    psd_path: str
    accepted_ownership: dict
    resource_failures: list[dict]
