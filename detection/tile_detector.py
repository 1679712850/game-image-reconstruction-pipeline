from detection.grouped_detector import scan_window
from tiling.tile_generator import generate_tiles
from taxonomy.categories import category_group
from taxonomy.planner import groups_for_window
import numpy as np


def detect_tiles(image, categories, infer, collect, scans, config, *, budget=None, pass_id="regional_tiled", candidates=(), scene_categories=()):
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
            tile_categories = categories
            if pass_id == 'gap_fill':
                # update_remaining has whitened accepted pixels. Avoid rescanning resolved tiles.
                pixels = np.asarray(image.crop(window).resize((32,32)).convert('RGB'))
                if float((pixels.min(axis=2) < 245).mean()) < .01:
                    tiles[-1]['skip_reason'] = 'resolved_coverage'
                    continue
            if config.category_planner.enabled:
                allowed = groups_for_window(window,categories,candidates,scene_categories,
                                            config.category_planner.max_groups_per_tile)
                explicit = set(scene_categories) if config.category_planner.preserve_scene_categories else set()
                tile_categories = [c for c in categories if c in explicit or category_group(c) in allowed]
                tiles[-1]['explicit_categories_preserved'] = sorted(explicit)
                tiles[-1]['category_groups'] = allowed
                tiles[-1]['allocation_reason'] = 'global_candidates_then_scene_categories'
            scan_window(image, window, tile_categories, infer, collect, scans, source="tile", tile_id=tile_id,
                        group_size=min(config.prompt_group_size, config.budget.max_categories_per_pass),
                        budget=budget, pass_id=pass_id, scale=size)
    return tiles
