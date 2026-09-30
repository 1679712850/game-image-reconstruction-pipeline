"""Agentic scene-understanding node with an injected VLM."""
from collections.abc import Callable

from agent.state import SceneState
from services.vlm_service import VLMService


def make_analyze_scene(service: VLMService) -> Callable[[SceneState], dict]:
    """Bind a service once; the resulting node accepts only state."""
    def analyze_scene(state: SceneState) -> dict:
        """Return validated semantics and projection."""
        analysis = service.analyze_scene(state["source_path"])
        return {"scene_analysis": analysis.model_dump(mode="json")}
    return analyze_scene
