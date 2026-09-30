"""Adapter-neutral prompt scheduling, with each failed scan isolated."""
import logging
from taxonomy.categories import category_group
from taxonomy.prompt_groups import groups_for

TILE_INSTRUCTION = (
    "This image is a local tile of a larger 2D / 2.5D isometric game map. "
    "Find every independently reusable environmental object in the requested group, "
    "including tiny, occluded and edge-truncated objects and building attachments. "
    "Include uncertain candidates for later review. Inspect under eaves, roofs, "
    "stairs, bridges and trees for lanterns, flags, pots, signs, pillars, stones, "
    "plants, tombstones, incense burners, talismans and cultivation props."
)


def grouped_categories(categories, size):
    groups = groups_for(categories)
    result = []
    for name, members in groups:
        if name == "other":
            by_group = {}
            for member in members:
                by_group.setdefault(category_group(member), []).append(member)
        else:
            by_group = {name: members}
        for group, values in by_group.items():
            result.extend((group, values[i:i+size]) for i in range(0, len(values), size))
    return result


def scan_window(image, window, categories, infer, collect, scans, *, source, tile_id=None,
                group_size=6, parent_id=None):
    for group_name, group in grouped_categories(categories, group_size):
        context = {"source": source, "tile_id": tile_id, "window": list(window),
                   "group": group_name, "categories": group, "parent_id": parent_id,
                   "instruction": TILE_INSTRUCTION if source != "global" else "Find whole scene objects and terrain with global context."}
        try:
            found = infer(image.crop(window), group, context)
            scan = {**context, "status": "ok", "candidates": len(found)}
            for item in found:
                collect(item, context)
        except Exception as error:
            scan = {**context, "status": "failed", "candidates": 0, "error": f"{type(error).__name__}: {error}"}
            logging.getLogger(__name__).warning("[%s] tile=%s group=%s failed: %s", "GlobalDetection" if source == "global" else "TileDetection", tile_id, group_name, error)
        scans.append(scan)
        logging.getLogger(__name__).info("[%s] tile=%s group=%s objects=%s", "GlobalDetection" if source == "global" else "TileDetection", tile_id, group_name, scan["candidates"])
