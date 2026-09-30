"""Bounded deterministic routing after every QA pass."""
from agent.state import SceneState


def route_after_qa(state: SceneState) -> str:
    """Retry only while failures exist and the configured budget remains."""
    if state.get("failed_objects") and state.get("retry_count", 0) < state.get("max_retry", 1):
        return "retry"
    return "continue"
