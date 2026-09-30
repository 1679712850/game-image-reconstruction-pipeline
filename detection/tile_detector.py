from detection.grouped_detector import scan_window
from tiling.tile_generator import generate_tiles


def detect_tiles(image, categories, infer, collect, scans, config):
    scales = list(dict.fromkeys([config.tiling.tile_size, *(config.multi_scale.scales if config.multi_scale.enabled else [])]))
    tiles, seen = [], set()
    for size in scales:
        for tile in generate_tiles(image.width, image.height, size, config.tiling.overlap, scale=size):
            window = (tile.x, tile.y, tile.x+tile.width, tile.y+tile.height)
            if window in seen:
                continue
            seen.add(window)
            tile_id = f"s{size}_{tile.tile_id}"
            tiles.append({**tile.as_dict(), "tile_id": tile_id})
            scan_window(image, window, categories, infer, collect, scans, source="tile", tile_id=tile_id,
                        group_size=config.prompt_group_size)
    return tiles
