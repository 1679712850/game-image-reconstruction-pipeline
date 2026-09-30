"""Fixed LangGraph backbone with injected services and optional checkpoints."""
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.routers import route_after_qa
from agent.state import SceneState
from app.config import PipelineConfig
from nodes.analyze_scene import make_analyze_scene
from nodes.build_metadata import build_metadata
from nodes.crop_objects import make_crop_objects
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
    adapters = services or ServiceBundle.create(mock=options.mock)
    taxonomy = categories_path or Path(__file__).resolve().parents[1] / "config" / "categories.yaml"

    def initialize(state: SceneState) -> dict:
        updates = load_image(state)
        updates.update(retry_count=state.get("retry_count", 0), max_retry=state.get("max_retry", options.max_retry))
        if updates["retry_count"] < 0 or updates["max_retry"] < 0:
            raise ValueError("Retry counters must be nonnegative")
        return updates

    nodes = {
        "load_image": initialize,
        "analyze_scene": make_analyze_scene(adapters.vlm),
        "plan_layers": make_plan_layers(taxonomy),
        "detect_instances": make_detect_instances(adapters.grounding, options.exercise_retry),
        "segment_instances": make_segment_instances(adapters.sam),
        "refine_masks": make_refine_masks(options.crop.alpha_threshold),
        "crop_objects": make_crop_objects(options.crop),
        "qa_objects": make_qa_objects(options),
        "retry_objects": make_retry_objects(options, adapters.sam),
        "upscale_objects": make_upscale_objects(adapters.upscale, options.upscale.enabled),
        "build_metadata": build_metadata,
        "reconstruct_scene": make_reconstruct_scene(options.reconstruction.enabled),
        "export": make_export(options.mock),
    }
    builder = StateGraph(SceneState)
    for name, node in nodes.items():
        builder.add_node(name, _observed(name, node, progress))

    main_path = [
        "load_image", "analyze_scene", "plan_layers", "detect_instances",
        "segment_instances", "refine_masks", "crop_objects", "qa_objects",
    ]
    builder.add_edge(START, main_path[0])
    for source, target in zip(main_path, main_path[1:]):
        builder.add_edge(source, target)
    builder.add_conditional_edges("qa_objects", route_after_qa, {
        "retry": "retry_objects", "continue": "upscale_objects",
    })
    builder.add_edge("retry_objects", "qa_objects")
    for source, target in (
        ("upscale_objects", "build_metadata"),
        ("build_metadata", "reconstruct_scene"),
        ("reconstruct_scene", "export"),
        ("export", END),
    ):
        builder.add_edge(source, target)
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
