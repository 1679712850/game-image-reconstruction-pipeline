"""Optional scene-level RGBA decomposition; instance detections remain separate."""
from collections.abc import Callable
from pathlib import Path

from agent.state import SceneState
from services.qwen_layered_service import QwenLayeredService


def make_decompose_layers(service: QwenLayeredService) -> Callable[[SceneState], dict]:
    """Bind the service outside serializable LangGraph state."""
    def decompose_layers(state: SceneState) -> dict:
        """Write model layers as auxiliary scene assets for review/export."""
        layers = service.decompose_layers(
            state["source_path"], output_dir=Path(state["output_dir"]) / "layers",
        )
        return {"decomposed_layers": layers}
    return decompose_layers
