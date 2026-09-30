"""Tile boundaries with overlap shading and failed-window labels."""
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from diagnostics.bbox_visualizer import open_image


def draw_tiles(source, output_path, tiles, failed_ids=()):
    image = open_image(source)
    depth = np.zeros((image.height, image.width), dtype=np.uint16)
    for tile in tiles:
        x, y = tile["x"], tile["y"]
        depth[y:y+tile["height"], x:x+tile["width"]] += 1
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[:, :, :3] = (20, 180, 255)
    rgba[:, :, 3] = np.minimum(np.maximum(depth.astype(int)-1, 0)*18, 120).astype(np.uint8)
    image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba))
    draw = ImageDraw.Draw(image)
    for tile in tiles:
        x, y, w, h = tile["x"], tile["y"], tile["width"], tile["height"]
        color = "red" if tile["tile_id"] in failed_ids else "cyan"
        draw.rectangle((x, y, x+w-1, y+h-1), outline=color, width=max(1, image.width//700))
        draw.text((x+3, y+3), tile["tile_id"], fill=color, stroke_width=1, stroke_fill="black")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
