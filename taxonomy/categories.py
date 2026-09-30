"""Fine-grained game-scene taxonomy and group metadata."""

CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "terrain": ("ground", "soil", "grass_ground", "stone_ground", "sand", "snow", "mud", "lava_ground", "cliff", "slope", "platform", "floating_island"),
    "vegetation": ("tree", "dead_tree", "bamboo", "shrub", "bush", "grass", "flower", "vine", "mushroom", "lotus", "spirit_plant", "herb"),
    "rock": ("mountain", "cliff_rock", "large_rock", "medium_rock", "small_rock", "rock_cluster", "floating_rock", "crystal_rock", "ore", "spirit_stone"),
    "structure": ("building", "house", "temple", "tower", "gate", "wall", "fence", "bridge", "stairs", "platform", "pillar", "stone_pillar", "wood_pillar", "railing", "roof", "door", "window", "arch", "ruin", "foundation"),
    "prop": ("barrel", "box", "crate", "chest", "table", "chair", "bench", "cart", "sign", "lamp", "lantern", "stone_lantern", "torch", "pot", "jar", "basket", "rack", "weapon_rack", "well", "flag", "boundary_marker", "wood_stake", "statue", "mechanism", "roof_decoration"),
    "cultivation_prop": ("alchemy_furnace", "forge", "formation", "formation_core", "formation_flag", "talisman", "spirit_stone", "spirit_crystal", "flying_sword", "magic_artifact", "altar", "totem", "teleport_array", "spirit_node", "incense_burner"),
    "graveyard": ("grave", "gravestone", "tombstone", "coffin", "burial_mound", "bone", "skeleton", "memorial_tablet"),
    "water": ("river", "lake", "pond", "waterfall", "stream", "swamp", "puddle"),
    "road": ("road", "stone_path", "dirt_path", "bridge_path", "stairs_path"),
    "fx_environment": ("fog", "cloud", "smoke", "fire", "spirit_energy", "portal", "light_beam", "magic_effect"),
}


def category_group(category: str) -> str:
    key = category.strip().lower().replace(" ", "_")
    if key == "spirit_stone":
        return "cultivation_prop"
    if key in {"rock", "rock_debris"}:
        return "rock"
    if key == "water":
        return "water"
    return next((group for group, values in CATEGORY_GROUPS.items() if key in values), "other")


def category_aliases(category: str) -> list[str]:
    return [category, category.replace("_", " ")]
