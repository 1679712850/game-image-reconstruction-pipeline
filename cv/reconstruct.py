"""Recompose original-sized assets with proper alpha blending and Y-sort."""
from pathlib import Path

from PIL import Image

from app.paths import read_rgba
from cv.layer_order import ordered_layers


def reconstruct_scene(
    width: int, height: int, objects: list[dict], output_path: str | Path,
) -> str:
    """Paste each original-resolution crop at crop_bbox.x/y, back to front."""
    if width <= 0 or height <= 0:
        raise ValueError("Scene dimensions must be positive")
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for obj in ordered_layers(objects):
        path = obj.get("accepted_asset") or obj.get("asset_path")
        box = obj.get("placement", {}).get("crop_bbox") or obj.get("crop_bbox")
        if not path or not box:
            continue  # An empty-mask manual-review record has no asset.
        asset = read_rgba(path)
        if asset.size != (box["w"], box["h"]):
            raise ValueError(f"Asset dimensions do not match crop_bbox: {path}")
        if box["x"] < 0 or box["y"] < 0 or box["x"] + box["w"] > width or box["y"] + box["h"] > height:
            raise ValueError(f"Crop lies outside the scene: {obj.get('id')}")
        canvas.alpha_composite(asset, (box["x"], box["y"]))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, "PNG")
    return str(target.resolve())
