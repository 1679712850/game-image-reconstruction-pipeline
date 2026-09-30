"""Command-line entry point for the scene reconstruction workflow."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agent.graph import build_graph
from agent.state import SceneState
from app.config import PipelineConfig, load_config


def parse_args() -> argparse.Namespace:
    """Parse input/output locations and explicit mock overrides."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Agentic 2D Game Scene Reconstruction Pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to output/<input stem>")
    parser.add_argument("--config", type=Path, default=root / "config" / "pipeline.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", dest="mock", action="store_true")
    mode.add_argument("--real", dest="mock", action="store_false")
    parser.set_defaults(mock=None)
    parser.add_argument("--max-retry", type=int)
    parser.add_argument("--exercise-retry", action="store_true", help="Inject one low-confidence mock detection")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> SceneState:
    """Invoke the graph and save a serializable execution record after END."""
    data = load_config(args.config).model_dump()
    if args.mock is not None:
        data["mock"] = args.mock
    if args.max_retry is not None:
        data["max_retry"] = args.max_retry
    if args.exercise_retry:
        data["exercise_retry"] = True
    config = PipelineConfig.model_validate(data)
    root = Path(__file__).resolve().parent
    output = (args.output or root / "output" / args.input.stem).expanduser().resolve()
    events = []
    stages = [
        ("load_image", "Load image"), ("analyze_scene", "Analyze scene"),
        ("plan_layers", "Plan layers"), ("detect_instances", "Detect objects"),
        ("segment_instances", "Segment objects"), ("refine_masks", "Refine masks"),
        ("crop_objects", "Crop assets"), ("qa_objects", "QA"),
        ("upscale_objects", "Upscale"), ("build_metadata", "Build metadata"),
        ("reconstruct_scene", "Reconstruct"), ("export", "Export"),
    ]
    labels = {name: (index, label) for index, (name, label) in enumerate(stages, 1)}

    def progress(name: str) -> None:
        events.append(name)
        if name == "retry_objects":
            print("[retry] Repair failed objects", flush=True)
        else:
            index, label = labels[name]
            print(f"[{index}/{len(stages)}] {label}", flush=True)

    graph = build_graph(config, progress=progress)
    initial: SceneState = {
        "source_path": str(args.input.expanduser().resolve()),
        "output_dir": str(output), "retry_count": 0,
        "max_retry": config.max_retry, "failed_objects": [],
    }
    result = graph.invoke(initial, config={"recursion_limit": 20 + 2 * config.max_retry})
    report = {"completed": True, "visited_nodes": events, "state": result}
    (output / "debug" / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Done. Reached END. Objects: {len(result['objects'])}; retries: {result['retry_count']}")
    if result["failed_objects"]:
        print(f"Manual review: {', '.join(result['failed_objects'])}")
    print(f"Scene: {result['scene_json']}")
    return result


def main() -> int:
    """Return a nonzero exit code for unsupported real adapters or bad inputs."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    try:
        run_pipeline(args)
    except (OSError, ValueError, NotImplementedError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
