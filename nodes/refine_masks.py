"""Conservative CV refinement of saved masks."""
from collections.abc import Callable

from agent.state import SceneState
from app.objects import refine_records


def make_refine_masks(threshold: int) -> Callable[[SceneState], dict]:
    """Bind the configured alpha threshold."""
    def refine_masks(state: SceneState) -> dict:
        """Normalize alpha noise and return copied object records."""
        return {"objects": refine_records(state["objects"], threshold)}
    return refine_masks
