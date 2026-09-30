"""Fixed LangGraph backbone with injected services and optional checkpoints."""
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.routers import route_after_qa, route_after_scene_qa
from agent.state import SceneState
from app.config import PipelineConfig
from nodes.analyze_scene import make_analyze_scene
from nodes.build_metadata import build_metadata
from nodes.crop_objects import make_crop_objects
from nodes.complete_objects import make_complete_objects
from nodes.decompose_layers import make_decompose_layers
from nodes.detect_instances import make_detect_instances
from nodes.export import make_export
from nodes.load_image import load_image
from nodes.plan_layers import make_plan_layers
from nodes.qa_objects import make_qa_objects
from nodes.reconstruct_scene import make_reconstruct_scene
from nodes.refine_masks import make_refine_masks
from nodes.retry_objects import make_retry_objects
from nodes.segment_instances import make_segment_instances
from nodes.upscale_objects import make_upscale_objects
from services.runtime import ServiceBundle
from services.scene_review_service import SceneReviewService
from nodes.scene_loop import make_update_remaining, make_qa_scene
from nodes.assign_ownership import make_assign_ownership
from nodes.p1_scene import make_p1_scene


def _observed(
    name: str, node: Callable[[SceneState], dict],
    progress: Callable[[str], None] | None,
) -> Callable[[SceneState], dict]:
    """Add optional per-invocation logging without global or serialized state."""
    def invoke(state: SceneState) -> dict:
        if progress is not None:
            progress(name)
        return node(state)
    return invoke


def build_graph(
    config: PipelineConfig | None = None,
    services: ServiceBundle | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    categories_path: Path | None = None,
) -> CompiledStateGraph:
    """Compile the real StateGraph; closures keep services out of checkpoints."""
    options = config or PipelineConfig()
    adapters = services or ServiceBundle.create(mock=options.mock, reviewer_backend=options.scene_loop.reviewer)
    reviewer = adapters.reviewer or SceneReviewService(options.scene_loop.reviewer)
    if options.scene_loop.enabled or options.p1.enabled:
        if not options.mock and reviewer.backend != options.scene_loop.reviewer:
            raise ValueError("Injected scene reviewer backend differs from scene_loop.reviewer")
        reviewer.validate_ready()
    taxonomy = categories_path or Path(__file__).resolve().parents[1] / "config" / "categories.yaml"
    if options.layer_decomposition.enabled:
        adapters.layered.validate_ready()
    if options.object_completion.enabled:
        adapters.image_edit.validate_ready()

    def initialize(state: SceneState) -> dict:
        updates = load_image(state)
        updates.update(retry_count=state.get("retry_count", 0), max_retry=state.get("max_retry", options.max_retry))
        updates.update(detection_round=0, archived_objects=[], scene_history=[],
                       scene_coverage=0.0, scene_no_progress=0, total_retry_count=0,
                       all_detections=[], detection_runs=[], working_path=updates["source_path"], coverage_mask_path="",
                       scene_next_categories=[], scene_continue=False, scene_stop_reason="")
        if updates["retry_count"] < 0 or updates["max_retry"] < 0:
            raise ValueError("Retry counters must be nonnegative")
        if state.get("edit_requests") and not options.object_completion.enabled:
            raise ValueError("edit_requests requires object_completion.enabled")
        return updates

    nodes = {
        "load_image": initialize,
        "analyze_scene": make_analyze_scene(adapters.vlm),
        "plan_layers": make_plan_layers(taxonomy),
        "detect_instances": make_detect_instances(adapters.grounding, options.exercise_retry, options.scene_loop if options.scene_loop.enabled else None, options.detection),
        "segment_instances": make_segment_instances(adapters.sam, options.detection, options.p1),
        "refine_masks": make_refine_masks(options.crop.alpha_threshold),
        "crop_objects": make_crop_objects(options.crop),
        "qa_objects": make_qa_objects(options),
        "retry_objects": make_retry_objects(options, adapters.sam, reviewer, adapters.grounding),
        "upscale_objects": make_upscale_objects(adapters.upscale, options.upscale.enabled),
        "build_metadata": build_metadata,
        "reconstruct_scene": make_reconstruct_scene(options.reconstruction.enabled, options.p1),
        "export": make_export(options.mock, {
            **adapters.provenance(
                layered_enabled=options.layer_decomposition.enabled,
                image_edit_enabled=options.object_completion.enabled,
            ),
            **({"upscale": "disabled"} if not options.upscale.enabled else {}),
        }, diagnostics_enabled=options.detection.diagnostics.enabled),
    }
    if options.p1.enabled:
        nodes['p1_scene'] = make_p1_scene(options, adapters.grounding, adapters.sam, reviewer)
        nodes['assign_ownership'] = make_assign_ownership(options)
    if options.layer_decomposition.enabled:
        nodes["decompose_layers"] = make_decompose_layers(adapters.layered)
    if options.object_completion.enabled:
        nodes["complete_objects"] = make_complete_objects(adapters.image_edit)
    if options.scene_loop.enabled:
        nodes["update_remaining"] = make_update_remaining(options)
        nodes["qa_scene"] = make_qa_scene(options, reviewer)
    builder = StateGraph(SceneState)
    for name, node in nodes.items():
        builder.add_node(name, _observed(name, node, progress))

    main_path = [
        "load_image", "analyze_scene", "plan_layers", "detect_instances",
        "segment_instances", "refine_masks", "crop_objects", "qa_objects",
    ]
    if options.layer_decomposition.enabled:
        main_path.insert(main_path.index("detect_instances"), "decompose_layers")
    builder.add_edge(START, main_path[0])
    for source, target in zip(main_path, main_path[1:]):
        builder.add_edge(source, target)
    after_ownership = "complete_objects" if options.object_completion.enabled else "upscale_objects"
    after_scene = "p1_scene" if options.p1.enabled else after_ownership
    builder.add_conditional_edges("qa_objects", route_after_qa, {
        "retry": "retry_objects", "continue": "update_remaining" if options.scene_loop.enabled else after_scene,
    })
    builder.add_edge("retry_objects", "qa_objects")
    if options.scene_loop.enabled:
        builder.add_edge("update_remaining", "qa_scene")
        builder.add_conditional_edges("qa_scene", route_after_scene_qa, {"detect": "detect_instances", "continue": after_scene})
    if options.object_completion.enabled:
        builder.add_edge("complete_objects", "upscale_objects")
    if options.p1.enabled:
        builder.add_edge("p1_scene", "assign_ownership")
        builder.add_edge("assign_ownership", after_ownership)
    for source, target in (
        ("upscale_objects", "build_metadata"),
        ("build_metadata", "reconstruct_scene"),
        ("reconstruct_scene", "export"),
        ("export", END),
    ):
        builder.add_edge(source, target)
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    return graph.with_config(recursion_limit=20 + options.scene_loop.max_rounds * (10 + 2 * options.max_retry))
