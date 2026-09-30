"""Reusable object operations for the main pass and targeted retries."""
from pathlib import Path
import numpy as np

from app.config import CropConfig
from app.paths import read_rgba
from cv.bbox import get_tight_bbox
from cv.crop import crop_rgba_by_mask
from cv.mask import read_mask, refine_mask, save_mask
from schemas.object import BBox, SceneObject
from services.sam_service import SAMService
from services.model_support import ModelUnavailableError


def segment_records(
    source_path: str, records: list[dict], service: SAMService, root: Path,
    *, local: bool = False, coverage_path: str | None = None, version: str = "",
) -> list[dict]:
    """Persist service masks before putting object records into graph state."""
    segment = service.segment_local if local else service.segment
    try:
        outputs = segment(source_path, records)
    except ModelUnavailableError:
        raise
    except (RuntimeError, ValueError, OSError):
        # Isolate an inference failure, retaining a review record for every input.
        outputs = []
        for record in records:
            try:
                outputs.extend(segment(source_path, [record]))
            except ModelUnavailableError:
                raise
            except (RuntimeError, ValueError, OSError) as error:
                outputs.append({**record, "mask": None, "error": f"segmentation_failed: {error}"})
    covered = read_mask(coverage_path) > 0 if coverage_path else None
    if len(outputs) != len(records):
        raise ValueError("Segmentation must return one record per input detection")
    if {item["id"] for item in outputs} != {item["id"] for item in records}:
        raise ValueError("Segmentation must preserve object IDs")
    objects = []
    for result in outputs:
        data = {key: value for key, value in result.items() if key != "mask"}
        obj = SceneObject.model_validate(data)
        if result.get("mask") is None:
            obj.mask_path = None
            obj.status = "retry"
            obj.error = obj.error or "segmentation_failed: missing mask"
            objects.append(obj.model_dump(mode="json"))
            continue
        mask = np.array(result["mask"], copy=True)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            obj.mask_path = None
            obj.status = "retry"
            obj.error = "segmentation_failed: invalid mask dimensions or non-finite mask"
            objects.append(obj.model_dump(mode="json"))
            continue
        if covered is not None:
            if covered.shape != mask.shape:
                raise ValueError("Coverage mask differs from segmentation size")
            mask[covered] = 0
        obj.mask_path = save_mask(mask, root / "masks" / f"{obj.id}{version}.png")
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
    *, version: str = "",
) -> list[dict]:
    """Crop object assets; empty or invalid masks remain reviewable records."""
    source = read_rgba(source_path)
    objects = []
    for record in records:
        obj = SceneObject.model_validate(record)
        obj.asset_path = None
        obj.crop_bbox = None
        obj.error = obj.error if not obj.mask_path else None
        if not obj.mask_path:
            obj.error = obj.error or "missing mask"
        else:
            mask = read_mask(obj.mask_path)
            if mask.shape != (source.height, source.width):
                obj.error = "mask dimensions do not match the source"
            else:
                box = get_tight_bbox(mask, config.alpha_threshold, config.padding)
                if box is None:
                    obj.error = "empty mask"
                else:
                    path = output_dir / ("effects" if obj.group == "fx_environment" else "assets") / f"{obj.id}{version}.png"
                    crop_rgba_by_mask(source, mask, box, path, threshold=config.alpha_threshold)
                    obj.crop_bbox = BBox(**box)
                    obj.asset_path = str(path.resolve())
        obj.status = "retry" if obj.error else "cropped"
        objects.append(SceneObject.model_validate(obj.model_dump()).model_dump(mode="json"))
    return objects
