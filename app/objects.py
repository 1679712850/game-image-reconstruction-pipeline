"""Reusable object operations for the main pass and targeted retries."""
from pathlib import Path

from app.config import CropConfig
from app.paths import read_rgba
from cv.bbox import get_tight_bbox
from cv.crop import crop_rgba_by_mask
from cv.mask import read_mask, refine_mask, save_mask
from schemas.object import BBox, SceneObject
from services.sam_service import SAMService


def segment_records(
    source_path: str, records: list[dict], service: SAMService, root: Path,
) -> list[dict]:
    """Persist service masks before putting object records into graph state."""
    outputs = service.segment(source_path, records)
    if len(outputs) != len(records):
        raise ValueError("Segmentation must return one record per input detection")
    if {item["id"] for item in outputs} != {item["id"] for item in records}:
        raise ValueError("Segmentation must preserve object IDs")
    objects = []
    for result in outputs:
        data = {key: value for key, value in result.items() if key != "mask"}
        obj = SceneObject.model_validate(data)
        obj.mask_path = save_mask(result["mask"], root / "masks" / f"{obj.id}.png")
        obj.status = "segmented"
        obj.error = None
        objects.append(obj.model_dump(mode="json"))
    return objects


def refine_records(records: list[dict], threshold: int) -> list[dict]:
    """Normalize grayscale masks without silently changing scene geometry."""
    objects = []
    for record in records:
        obj = SceneObject.model_validate(record)
        if obj.mask_path:
            mask = refine_mask(read_mask(obj.mask_path), threshold=threshold)
            save_mask(mask, obj.mask_path)
        objects.append(obj.model_dump(mode="json"))
    return objects


def crop_records(
    source_path: str, records: list[dict], output_dir: Path, config: CropConfig,
) -> list[dict]:
    """Crop object assets; empty or invalid masks remain reviewable records."""
    source = read_rgba(source_path)
    objects = []
    for record in records:
        obj = SceneObject.model_validate(record)
        obj.asset_path = None
        obj.crop_bbox = None
        obj.error = None
        if not obj.mask_path:
            obj.error = "missing mask"
        else:
            mask = read_mask(obj.mask_path)
            if mask.shape != (source.height, source.width):
                obj.error = "mask dimensions do not match the source"
            else:
                box = get_tight_bbox(mask, config.alpha_threshold, config.padding)
                if box is None:
                    obj.error = "empty mask"
                else:
                    path = output_dir / "assets" / f"{obj.id}.png"
                    crop_rgba_by_mask(source, mask, box, path, threshold=config.alpha_threshold)
                    obj.crop_bbox = BBox(**box)
                    obj.asset_path = str(path.resolve())
        obj.status = "retry" if obj.error else "cropped"
        objects.append(SceneObject.model_validate(obj.model_dump()).model_dump(mode="json"))
    return objects
