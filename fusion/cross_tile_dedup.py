"""Conservative identity evidence; close same-class neighbours stay separate."""
import math
import numpy as np


def geometry(a, b):
    ax, ay, ar, ab = a.bbox
    bx, by, br, bb = b.bbox
    intersection = max(0, min(ar, br)-max(ax, bx)) * max(0, min(ab, bb)-max(ay, by))
    union = a.area + b.area - intersection
    iou = intersection / union if union else 0
    overlap = intersection / min(a.area, b.area) if min(a.area, b.area) else 0
    distance = math.hypot((ax+ar-bx-br)/2, (ay+ab-by-bb)/2)
    scale = max(1, min(math.hypot(ar-ax, ab-ay), math.hypot(br-bx, bb-by)))
    ratio = min(a.area, b.area) / max(a.area, b.area, 1)
    return iou, overlap, distance/scale, ratio


def appearance_similarity(a, b):
    if not a.appearance or not b.appearance:
        return None
    aa, bb = np.array(a.appearance), np.array(b.appearance)
    return float(np.minimum(aa, bb).sum() / max(aa.sum(), bb.sum(), 1e-9))


def same_object(a, b, config):
    if a.category != b.category:
        return False
    iou, overlap, distance, ratio = geometry(a, b)
    if not iou:
        return False
    if (a.parent_id in [b.id, *b.merged_from] or b.parent_id in [a.id, *a.merged_from]) and overlap >= config.overlap_ratio and distance <= 1.25 and ratio >= .1:
        return True
    # A small prop inside a large same-class region is not automatically a duplicate.
    if iou >= config.bbox_iou and distance <= config.center_distance and ratio >= .35:
        return True
    if overlap >= config.overlap_ratio and distance <= config.center_distance and ratio >= .5:
        return True
    # Partially visible instances need independent-window + appearance evidence.
    distinct_window = a.window != b.window
    complementary = bool(set(a.truncated_edges) & {"left", "top"}) != bool(set(b.truncated_edges) & {"left", "top"})
    similarity = appearance_similarity(a, b)
    return bool(distinct_window and (a.is_truncated or b.is_truncated)
                and overlap >= config.overlap_ratio and distance <= 1.25
                and (complementary or not a.is_truncated or not b.is_truncated)
                and similarity is not None and similarity >= config.appearance_similarity)
