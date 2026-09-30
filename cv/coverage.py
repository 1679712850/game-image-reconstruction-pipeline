"""Pixel-union coverage and immutable residual images, independent of LangGraph."""
from pathlib import Path

import numpy as np
from PIL import Image

from app.paths import read_rgba
from cv.mask import read_mask, save_mask
from cv.tiles import tile_windows


def mask_union(objects: list[dict], size: tuple[int, int], threshold: int, accepted_only: bool = False) -> np.ndarray:
    """Count overlap once; empty/error candidates never contribute to coverage."""
    union = np.zeros((size[1], size[0]), dtype=bool)
    for obj in objects:
        if obj.get("error") or not obj.get("asset_path") or not obj.get("mask_path"):
            continue
        if accepted_only and obj.get("status") != "pass":
            continue
        mask = read_mask(obj["mask_path"])
        if mask.shape != union.shape:
            raise ValueError("Coverage mask dimensions differ from source")
        union |= mask > threshold
    return union


def write_remaining(source_path: str, objects: list[dict], directory: Path, threshold: int) -> dict:
    """Whiten accepted mask pixels only; keep source RGB and alpha unchanged on disk."""
    source = read_rgba(source_path)
    rgba = np.array(source)
    eligible = rgba[:, :, 3] > threshold
    accepted = mask_union(objects, source.size, threshold, True) & eligible
    candidates = mask_union(objects, source.size, threshold) & eligible
    denominator = int(eligible.sum())
    remaining = rgba[:, :, :3].copy()
    remaining[accepted | ~eligible] = 255
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "remaining.png"
    Image.fromarray(remaining).save(path)
    coverage_path = save_mask(accepted, directory / "coverage.png")
    regions = []
    for x0, y0, x1, y1 in tile_windows(source.width, source.height, 512, 0):
        valid = eligible[y0:y1, x0:x1]
        count = int(valid.sum())
        regions.append({"bbox": {"x": x0, "y": y0, "w": x1-x0, "h": y1-y0},
                        "uncovered_fraction": float((valid & ~accepted[y0:y1, x0:x1]).sum() / count) if count else 0.0})
    return {"working_path": str(path.resolve()), "coverage_mask_path": coverage_path,
            "accepted_coverage": float(accepted.sum() / denominator) if denominator else 0.0,
            "candidate_coverage": float(candidates.sum() / denominator) if denominator else 0.0,
            "eligible_pixels": denominator, "uncovered_regions": sorted(regions, key=lambda x: -x["uncovered_fraction"])[:16]}
