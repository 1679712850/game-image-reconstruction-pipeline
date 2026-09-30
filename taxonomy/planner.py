"""Region allocation from global candidates, scene categories and neighboring evidence."""
from collections import Counter
from collections.abc import Sequence
from taxonomy.categories import category_group


def groups_for_window(window: tuple[int,int,int,int], categories: Sequence[str],
                      candidates: Sequence, scene_categories: Sequence[str], max_groups: int = 4) -> list[str]:
    """Prefer groups observed in/near the region; fall back to scene analysis.

    Group priorities are evidence-based, not inferred from the average pixel color.
    Unknown groups remain eligible through configured scene categories. A capped
    fallback is a recall trade-off recorded in tile diagnostics, never a claim
    that absent groups do not exist.
    """
    groups = list(dict.fromkeys(category_group(c) for c in categories))
    scores: Counter[str] = Counter()
    x,y,r,b = window
    pad = max(r-x,b-y)*.1
    for candidate in candidates:
        a,c,d,e = candidate.bbox
        if min(d,r+pad)>max(a,x-pad) and min(e,b+pad)>max(c,y-pad):
            scores[category_group(candidate.category)] += 2+candidate.confidence
    for category in scene_categories:
        scores[category_group(category)] += .1
    # Neighboring natural/architectural details are sensible bounded extensions.
    if scores['vegetation'] > 1:
        scores['rock'] += .5
    if scores['structure'] > 1:
        scores['prop'] += .5
    ranked = sorted(groups,key=lambda g:(-scores[g],groups.index(g)))
    return ranked[:max_groups]
