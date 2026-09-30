"""Explicit future node contracts; none is connected to the V1 graph."""
from agent.state import SceneState


def decompose_layers(state: SceneState) -> dict:
    """Future node using ServiceBundle.layered.decompose_layers."""
    raise NotImplementedError("TODO: Qwen-Image-Layered semantic RGBA decomposition")


def analyze_occlusion(state: SceneState) -> dict:
    """Future decision node for occlusion relationships."""
    raise NotImplementedError("TODO: CV/VLM occlusion analysis")


def complete_objects(state: SceneState) -> dict:
    """Future completion node using ServiceBundle.image_edit."""
    raise NotImplementedError("TODO: object completion and inpainting")


def export_psd(state: SceneState) -> dict:
    """Future optional call into exporters.psd_exporter."""
    raise NotImplementedError("TODO: PSD export node")


def export_godot(state: SceneState) -> dict:
    """Future optional call into exporters.godot_exporter."""
    raise NotImplementedError("TODO: Godot export node")
