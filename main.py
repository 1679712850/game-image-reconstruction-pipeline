"""Command-line entry point for the scene reconstruction workflow."""
import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from agent.graph import build_graph
from agent.state import SceneState
from app.config import PipelineConfig, load_config
from app.edits import load_edit_requests
from app.models import load_models
from services.runtime import ServiceBundle
from services.model_support import ModelUnavailableError


def parse_args() -> argparse.Namespace:
    """Parse input/output locations and explicit mock overrides."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Agentic 2D Game Scene Reconstruction Pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to output/<input stem>")
    parser.add_argument("--config", type=Path, default=root / "config" / "pipeline.yaml")
    parser.add_argument("--models-config", type=Path, default=root / "config" / "models.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override the real-model device")
    parser.add_argument("--offline", action="store_true", help="Load real models from local cache only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", dest="mock", action="store_true")
    mode.add_argument("--real", dest="mock", action="store_false")
    parser.set_defaults(mock=None)
    parser.add_argument("--max-retry", type=int)
    parser.add_argument("--max-rounds", type=int, help="Maximum bounded scene detection rounds")
    parser.add_argument("--scene-reviewer", choices=("rules", "llm"), help="Scene continuation policy")
    parser.add_argument("--exercise-retry", action="store_true", help="Inject one low-confidence mock detection")
    parser.add_argument("--decompose-layers", action="store_true", help="Enable optional Qwen layer generation (mock or local weights only)")
    parser.add_argument("--edit-requests", type=Path, help="JSON list of object_id, crop-space mask_path and prompt; enables edit candidates")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> SceneState:
    """Invoke the graph and save a serializable execution record after END."""
    data = load_config(args.config).model_dump()
    if args.mock is not None:
        data["mock"] = args.mock
    if args.max_retry is not None:
        data["max_retry"] = args.max_retry
    if args.max_rounds is not None:
        data["scene_loop"]["max_rounds"] = args.max_rounds
    if args.scene_reviewer is not None:
        data["scene_loop"]["reviewer"] = args.scene_reviewer
    if args.exercise_retry:
        data["exercise_retry"] = True
    if args.decompose_layers:
        data["layer_decomposition"]["enabled"] = True
    edit_requests = load_edit_requests(args.edit_requests) if args.edit_requests else []
    if args.edit_requests is not None:
        data["object_completion"]["enabled"] = True
    config = PipelineConfig.model_validate(data)
    root = Path(__file__).resolve().parent
    output = (args.output or root / "output" / args.input.stem).expanduser().resolve()
    events = []
    stages = [
        ("load_image", "Load image"), ("analyze_scene", "Analyze scene"),
        ("plan_layers", "Plan layers"), ("detect_instances", "Detect objects"),
        ("segment_instances", "Segment objects"), ("refine_masks", "Refine masks"),
        ("crop_objects", "Crop assets"), ("qa_objects", "QA"),
        ("update_remaining", "Update remaining scene"), ("qa_scene", "Scene coverage review"),
        ("upscale_objects", "Upscale"), ("build_metadata", "Build metadata"),
        ("reconstruct_scene", "Reconstruct"), ("export", "Export"),
    ]
    if config.layer_decomposition.enabled:
        stages.insert(3, ("decompose_layers", "Decompose RGBA layers"))
    if config.p1.enabled:
        index = next(i for i, item in enumerate(stages) if item[0] == "upscale_objects")
        stages[index:index] = [("p1_scene", "Review coverage and retry problem regions"),
                              ("assign_ownership", "Resolve visible ownership and complete terrain")]
    if config.object_completion.enabled:
        index = next(i for i, item in enumerate(stages) if item[0] == "upscale_objects")
        stages.insert(index, ("complete_objects", "Generate edit candidates"))
    labels = {name: (index, label) for index, (name, label) in enumerate(stages, 1)}

    def progress(name: str) -> None:
        events.append(name)
        if name == "retry_objects":
            print("[retry] Repair failed objects", flush=True)
        else:
            index, label = labels[name]
            print(f"[{index}/{len(stages)}] {label}", flush=True)

    models = load_models(args.models_config) if not config.mock else None
    if models is not None:
        overrides = {"local_files_only": True} if args.offline else {}
        if args.device is not None:
            overrides["device"] = args.device
        models = models.model_copy(update=overrides)
    services = ServiceBundle.create(mock=config.mock, models=models, reviewer_backend=config.scene_loop.reviewer)
    graph = build_graph(config, services=services, progress=progress)
    initial: SceneState = {
        "source_path": str(args.input.expanduser().resolve()),
        "output_dir": str(output), "retry_count": 0,
        "max_retry": config.max_retry, "failed_objects": [],
        "edit_requests": edit_requests,
    }
    result = graph.invoke(initial)
    report = {"completed": True, "visited_nodes": events, "state": result,
              "models": models.model_dump(mode="json") if models else None,
              "pipeline": config.model_dump(mode="json"),
              "backends": services.provenance(
                  layered_enabled=config.layer_decomposition.enabled,
                  image_edit_enabled=config.object_completion.enabled,
              )}
    (output / "debug" / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Done. Reached END. Objects: {len(result['objects'])}; retries: {result['retry_count']}")
    if result.get("scene_qa"):
        print(f"Scene rounds: {result['detection_round']}; accepted mask coverage: {result['scene_coverage']:.1%}; stop: {result['scene_stop_reason']}")
        print(f"Scene QA: {result['scene_qa']['status']}; reviewer: {result['scene_qa']['backend']}; {result['scene_qa']['decision']['reason']}")
    if result["failed_objects"]:
        print(f"Manual review: {', '.join(result['failed_objects'])}")
    print(f"Scene: {result['scene_json']}")
    return result


def main() -> int:
    """Return a nonzero exit code for unsupported real adapters or bad inputs."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        run_pipeline(args)
    except (OSError, ValueError, NotImplementedError, ModelUnavailableError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
