"""Remaining future contracts; Qwen nodes now live in their own modules."""
from agent.state import SceneState


def analyze_occlusion(state: SceneState) -> dict:
    """Future decision node for occlusion relationships."""
    raise NotImplementedError("TODO: CV/VLM occlusion analysis")


def export_psd(state: SceneState) -> dict:
    """Future optional call into exporters.psd_exporter."""
    raise NotImplementedError("TODO: PSD export node")


def export_godot(state: SceneState) -> dict:
    """Future optional call into exporters.godot_exporter."""
    raise NotImplementedError("TODO: Godot export node")
