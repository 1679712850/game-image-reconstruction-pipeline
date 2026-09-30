"""Deterministic semantic layer planning; no generative decomposition yet."""
from collections.abc import Callable
from pathlib import Path


from agent.state import SceneState


def make_plan_layers(categories_path: Path) -> Callable[[SceneState], dict]:
    """Load the category taxonomy once when constructing the graph."""
    try:
        import yaml
        with categories_path.open(encoding="utf-8") as stream:
            groups = yaml.safe_load(stream)["layers"]
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load category taxonomy") from error

    def plan_layers(state: SceneState) -> dict:
        """Group detected semantics into ordered layer descriptors."""
        categories = list(dict.fromkeys(state["scene_analysis"]["categories"]))
        layers, assigned = [], set()
        for group in groups:
            members = [category for category in categories if category in group["categories"]]
            if members:
                layers.append({"name": group["name"], "categories": members, "order": len(layers)})
                assigned.update(members)
        remainder = [category for category in categories if category not in assigned]
        if remainder:
            layers.append({"name": "other", "categories": remainder, "order": len(layers)})
        return {"layer_plan": layers}
    return plan_layers
