"""Explicit terrain semantics; ambiguous hybrid geometry stays reviewable."""
from collections import Counter
from taxonomy.categories import category_group


TERRAIN = {
    "grass": "grass", "grass_ground": "grass", "ground": "ground", "soil": "soil",
    "dirt_path": "road", "road": "road", "stone_road": "road", "stone_path": "road",
    "stone_ground": "stone_ground", "sand": "sand", "snow": "snow", "water": "water",
    "river": "water", "lake": "water", "pond": "water", "stream": "water", "puddle": "water",
    "swamp": "swamp", "mud": "soil", "lava": "lava", "lava_ground": "lava",
    "cliff_surface": "cliff_ground", "mountain_ground": "mountain_ground",
    "platform_floor": "platform_floor", "wooden_floor": "wooden_floor",
    "bridge_surface": "bridge_surface", "bridge_path": "bridge_surface",
    "shadow_ground": "shadow_ground", "fog_region": "fog_region",
}
HYBRID = {"cliff", "mountain", "large_ruin", "bridge", "river_bank", "wall", "platform", "floating_island"}
PRIORITY = {"base_terrain": 10, "terrain_detail": 20, "vegetation": 30,
            "building": 40, "large_object": 50, "small_object": 60,
            "character": 70, "foreground_effect": 80}


class SceneElementClassifier:
    def classify(self, record, size, repeats=1):
        category = record["category"].strip().lower().replace(" ", "_")
        group = category_group(category)
        area = record["bbox"]["w"] * record["bbox"]["h"] / (size[0] * size[1])
        kind, layer, reason = "instance", "large_object", "bounded entity semantics"
        if category in TERRAIN:
            kind, layer, reason = "terrain", "base_terrain", "continuous surface semantics"
        elif category in HYBRID:
            kind, layer, reason = "hybrid", "large_object", "base/structure split requires separate evidence"
        elif group == "fx_environment":
            kind, layer = "effect", "foreground_effect"
        elif category in {"npc", "monster", "character", "player"}:
            layer = "character"
        elif group == "vegetation":
            layer = "vegetation"
        elif category in {"building", "house", "temple", "tower"}:
            layer = "building"
        elif area < .01 or group in {"prop", "cultivation_prop", "graveyard"}:
            layer = "small_object"
        return {**record, "category": category, "group": group, "element_type": kind, "layer_group": layer,
                "requires_individual_export": kind != "terrain",
                "requires_inpainting": kind in {"terrain", "hybrid"},
                "classification_reason": f"{reason}; area_ratio={area:.5f}; repeats={repeats}",
                "uncertain": kind == "hybrid" or (group == "other" and category not in TERRAIN
                                                   and layer != "character" and not record.get('classification_confidence')),
                "hybrid_components": [f"{category}_base", f"{category}_structure"] if kind == "hybrid" else [],
                "ownership_priority": PRIORITY[layer]}

    def classify_all(self, records, size):
        counts = Counter(r["category"] for r in records)
        return [self.classify(r, size, counts[r["category"]]) for r in records]
