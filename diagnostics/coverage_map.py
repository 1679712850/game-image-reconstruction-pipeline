"""Separate scan coverage, tile overlap and candidate density channels."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from diagnostics.bbox_visualizer import open_image, xyxy


def coverage_arrays(size, scans, tiles, candidates):
    width, height = size
    coverage = np.zeros((height, width), dtype=np.uint16)
    tile_depth = np.zeros_like(coverage)
    density = np.zeros_like(coverage)
    for window in {tuple(s["window"]) for s in scans if s.get("status") == "ok"}:
        x, y, r, b = window
        coverage[y:b, x:r] += 1
    for tile in tiles:
        x, y = tile["x"], tile["y"]
        tile_depth[y:y+tile["height"], x:x+tile["width"]] += 1
    for candidate in candidates:
        box = xyxy(candidate)
        if box:
            x, y, r, b = box
            density[y:b, x:r] += 1
    return coverage, tile_depth, density


def draw_coverage(source, output_path, scans, tiles, candidates):
    image = open_image(source)
    coverage, depth, density = coverage_arrays(image.size, scans, tiles, candidates)
    overlay = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    overlay[coverage == 0] = (240, 40, 40)
    overlay[(coverage > 0) & (density == 0)] = (35, 90, 210)
    visible = density > 0
    overlay[visible, 0] = np.minimum(255, 70+np.log1p(density[visible])*65).astype(np.uint8)
    overlay[visible, 1] = 220
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], np.minimum(depth*15, 120).astype(np.uint8))
    result = Image.blend(image, Image.fromarray(overlay), .55)
    ImageDraw.Draw(result).text((4, 4), "RED unscanned | BLUE scanned/no candidates | YELLOW density | violet tile overlap", fill="white", stroke_width=1, stroke_fill="black")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return {"unscanned_pixels": int((coverage == 0).sum()),
            "scanned_without_candidates_pixels": int(((coverage > 0) & (density == 0)).sum()),
            "max_tile_overlap": int(depth.max())}
