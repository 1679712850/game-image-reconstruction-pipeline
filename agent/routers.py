"""Bounded deterministic routing after every QA pass."""
from agent.state import SceneState


def route_after_qa(state: SceneState) -> str:
    """Retry only while failures exist and the configured budget remains."""
    eligible = state.get('retryable_objects', state.get('failed_objects'))
    if eligible and state.get("retry_count", 0) < state.get("max_retry", 1):
        return "retry"
    return "continue"


def route_after_scene_qa(state: SceneState) -> str:
    """Only the already-budgeted scene decision may start another round."""
    return "detect" if state.get("scene_continue", False) else "continue"
