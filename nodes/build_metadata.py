"""Build logical coordinates independently of texture resolution."""
from agent.state import SceneState
from app.paths import read_rgba
from cv.alpha import alpha_mask
from cv.pivot import ground_pivot
from schemas.object import SceneObject, Pivot


def build_metadata(state: SceneState) -> dict:
    """Compute local pivots and global ground-y sort values."""
    objects = []
    for record in state.get("objects", []):
        obj = SceneObject.model_validate(record)
        if obj.asset_path and obj.crop_bbox:
            image = read_rgba(obj.asset_path)
            obj.pivot = Pivot(**ground_pivot(alpha_mask(image)))
            obj.z_order = obj.crop_bbox.y + obj.pivot.y
            obj.logical_size = image.size
            if obj.hd_asset_path:
                obj.texture_size = read_rgba(obj.hd_asset_path).size
            else:
                obj.texture_size = image.size
                obj.texture_scale = 1
        objects.append(obj.model_dump(mode="json"))
    return {"objects": objects}
