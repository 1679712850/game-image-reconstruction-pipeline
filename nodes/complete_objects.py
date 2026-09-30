"""Generate explicit edit candidates without replacing verified instance assets."""
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from agent.state import SceneState
from app.edits import validate_edit_requests
from schemas.generation import ObjectEditResult
from services.image_edit_service import ImageEditService


def make_complete_objects(service: ImageEditService) -> Callable[[SceneState], dict]:
    """Bind Qwen image editing; automatic occlusion diagnosis is not assumed."""
    def complete_objects(state: SceneState) -> dict:
        """Only process requested IDs, saving portable masks and candidate records."""
        requests = validate_edit_requests(state.get("edit_requests", []))
        objects = {obj["id"]: obj for obj in state.get("objects", [])}
        for request in requests:
            obj = objects.get(request.object_id)
            if obj is None or not obj.get("asset_path") or not obj.get("crop_bbox"):
                raise ValueError(f"Edit request references an unknown/uncropped object: {request.object_id}")
        root = Path(state["output_dir"])
        records = []
        for request in requests:
            obj = objects[request.object_id]
            asset = service.complete_object(
                obj["asset_path"], request.mask_path, prompt=request.prompt,
                output_path=root / "assets_edited" / f"{request.object_id}.png",
            )
            mask_path = root / "edit_masks" / f"{request.object_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(Path(request.mask_path)) as image:
                image.convert("L").save(mask_path, format="PNG")
            box = obj["crop_bbox"]
            records.append(ObjectEditResult(
                object_id=request.object_id, source_asset_path=obj["asset_path"],
                mask_path=str(mask_path.resolve()), asset_path=asset, prompt=request.prompt,
                crop_bbox=box, logical_size=(box["w"], box["h"]), mock=service.mock,
                status="mock_noop" if service.mock else "manual_review",
            ).model_dump(mode="json"))
        return {"object_edits": records}
    return complete_objects
