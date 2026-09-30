"""Canonical labels for model and human vocabulary."""
from .categories import category_group

CATEGORY_ALIASES: dict[str, str] = {
    "stone lamp": "stone_lantern", "stone lantern": "stone_lantern", "garden lantern": "stone_lantern", "temple lantern": "stone_lantern", "lantern pillar": "stone_lantern", "石灯": "stone_lantern", "石灯笼": "stone_lantern", "灯柱": "stone_lantern", "寺庙石灯": "stone_lantern",
    "flag": "flag", "banner": "flag", "war banner": "flag", "阵旗": "formation_flag", "旗帜": "flag",
    "column": "pillar", "stone column": "stone_pillar", "石柱": "stone_pillar",
    "gravestone": "tombstone", "grave marker": "tombstone", "墓碑": "tombstone",
    "spirit stone": "spirit_stone", "mana stone": "spirit_stone", "灵石": "spirit_stone",
    "wooden barrel": "barrel", "wooden box": "box", "treasure chest": "chest", "宝箱": "chest", "瓦罐": "jar", "路牌": "sign",
    "street light": "lamp", "street lamp": "lamp", "路灯": "lamp", "界碑": "boundary_marker", "木桩": "wood_stake",
    "栅栏": "fence", "小型岩石": "small_rock", "小岩石": "small_rock", "花草": "flower", "灌木": "shrub",
    "箱子": "box", "木桶": "barrel", "香炉": "incense_burner", "雕像": "statue", "武器架": "weapon_rack",
    "屋檐": "roof", "屋檐装饰": "roof_decoration", "台阶": "stairs", "悬浮石": "floating_rock", "浮空石": "floating_rock",
    "符箓": "talisman", "法器": "magic_artifact", "灵晶": "spirit_crystal", "棺材": "coffin", "机关": "mechanism",
    "wood stump": "wood_stake", "tree stump": "wood_stake", "flower pot": "pot", "花盆": "pot",
}


def normalize_category(label: str, *, default: str | None = None) -> str | None:
    key = " ".join(label.strip().lower().strip(". ").replace("_", " ").split())
    canonical = CATEGORY_ALIASES.get(key, key.replace(" ", "_"))
    return canonical if category_group(canonical) != "other" or canonical in {"tree", "rock", "building", "other"} else default
