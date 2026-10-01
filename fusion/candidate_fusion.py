"""Fuse observations while preserving original confidence and source lineage."""
from fusion.cross_tile_dedup import same_object
from fusion.spatial_index import CandidateIndex


def merge_candidates(a, b):
    # A complete observation is stronger geometric evidence than a clipped crop.
    winner = max((a, b), key=lambda c: (not c.is_truncated, c.confidence, c.area))
    merged = winner.model_copy(deep=True)
    if a.is_truncated and b.is_truncated:
        merged.bbox = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                       max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
        merged.truncated_edges = sorted(set(a.truncated_edges + b.truncated_edges))
        merged.is_truncated = True  # Only a complete re-observation clears this.
    merged.merged_from = list(dict.fromkeys([a.id, *a.merged_from, b.id, *b.merged_from]))
    merged.observations = list({o["id"]: o for o in [*a.observations, *b.observations]}.values())
    merged.aliases = list(dict.fromkeys([*a.aliases, *b.aliases]))
    # Repeated prompts on the same window are correlated, so add evidence once/window.
    windows = {}
    for observation in merged.observations:
        key = tuple(observation["window"])
        windows[key] = max(windows.get(key, 0), observation["confidence"])
    scores = sorted(windows.values(), reverse=True)
    merged.confidence = min(.99, scores[0] + .15 * sum(scores[1:])) if scores else winner.confidence
    merged.redetected = a.redetected or b.redetected
    merged.parent_id = a.parent_id or b.parent_id
    return merged


def fuse_candidates(candidates, config):
    kept, rejected = [], []
    spatial = CandidateIndex()
    for candidate in sorted(candidates, key=lambda c: (not c.is_truncated, c.confidence), reverse=True):
        match = next((i for i in spatial.query(candidate) if same_object(kept[i], candidate, config)), None)
        if match is None:
            kept.append(candidate.model_copy(deep=True))
            spatial.update(len(kept)-1, kept[-1])
        else:
            old = kept[match]
            merged = merge_candidates(old, candidate)
            loser = candidate if merged.id != candidate.id else old
            duplicate = loser.model_copy(deep=True)
            duplicate.reject_reason = "cross_tile_duplicate" if old.window != candidate.window else "duplicate"
            duplicate.parent_id = merged.id
            rejected.append(duplicate)
            kept[match] = merged
            spatial.update(match, merged)
    return kept, rejected
