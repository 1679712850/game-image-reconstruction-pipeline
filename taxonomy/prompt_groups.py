"""Focused prompts reduce competition between long-tail object classes."""
PROMPT_GROUPS: dict[str, tuple[str, ...]] = {
    "structure": ("building", "tower", "wall", "pillar", "stone_pillar", "stairs", "bridge", "gate", "railing", "roof", "door"),
    "vegetation": ("tree", "bush", "shrub", "grass", "flower", "vine", "bamboo", "dead_tree", "herb"),
    "small_props": ("stone_lantern", "lantern", "flag", "formation_flag", "barrel", "box", "jar", "sign", "bench", "incense_burner", "weapon_rack", "chest", "tombstone"),
    "cultivation": ("formation", "spirit_stone", "spirit_crystal", "altar", "flying_sword", "talisman", "alchemy_furnace", "magic_artifact", "spirit_node"),
    "terrain": ("ground", "rock", "small_rock", "mountain", "water", "road", "platform", "floating_rock"),
}


def groups_for(categories: list[str]) -> list[tuple[str, list[str]]]:
    result = []
    remaining = list(dict.fromkeys(categories))
    for name, group in PROMPT_GROUPS.items():
        members = [category for category in remaining if category in group]
        if members:
            result.append((name, members))
            remaining = [category for category in remaining if category not in members]
    if remaining:
        result.append(("other", remaining))
    return result
