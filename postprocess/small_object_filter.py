"""Size-aware thresholds and explicit review protection for small props."""
from taxonomy.categories import CATEGORY_GROUPS

SMALL_CATEGORIES = set(CATEGORY_GROUPS["prop"] + CATEGORY_GROUPS["cultivation_prop"] + CATEGORY_GROUPS["graveyard"]) | {
    "flower", "grass", "herb", "mushroom", "small_rock", "pillar", "stone_pillar", "wood_pillar", "shrub", "bush",
}


def threshold_for(candidate, image_area, config):
    ratio = candidate.area / image_area
    if ratio < config.small_area_ratio:
        return config.small_object
    if ratio >= config.large_area_ratio:
        return config.large_object
    return config.default


def protect_small(candidate) -> bool:
    return candidate.category in SMALL_CATEGORIES and any(
        o.get("source") in {"tile", "redetection"} for o in candidate.observations)
