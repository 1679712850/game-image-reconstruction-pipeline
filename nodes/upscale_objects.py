"""Upscale texture pixels without moving scene geometry."""
from collections.abc import Callable

from agent.state import SceneState
from app.paths import read_rgba
from services.upscale_service import UpscaleService, choose_scale


def make_upscale_objects(service: UpscaleService, enabled: bool) -> Callable[[SceneState], dict]:
    """Bind the upscaler once, allowing the optional stage to be disabled."""
    def upscale_objects(state: SceneState) -> dict:
        """Return distinct logical and texture dimensions for each asset."""
        objects = []
        for record in state.get("objects", []):
            obj = dict(record)
            if obj.get("asset_path"):
                size = read_rgba(obj["asset_path"]).size
                scale = choose_scale(*size) if enabled else 1
                hd = service.upscale(obj["asset_path"], scale) if enabled else None
                obj.update(
                    hd_asset_path=hd, logical_size=list(size),
                    texture_size=list(read_rgba(hd).size if hd else size),
                    texture_scale=scale,
                )
            objects.append(obj)
        return {"objects": objects}
    return upscale_objects
