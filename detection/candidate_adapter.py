"""Validate model observations without losing invalid/unknown outputs."""
import math
import numpy as np

from schemas.detection_candidate import DetectionCandidate
from taxonomy.aliases import normalize_category
from taxonomy.categories import category_group
from tiling.coordinate_mapper import restore_bbox
from postprocess.truncation_detector import truncated_edges


def observation_candidate(item, context, image, candidate_id, config, method, categories):
    label = str(item.get("category", ""))
    raw_label = str(item.get("raw_label", label))
    category = normalize_category(label)
    allowed = {normalize_category(c, default=c.replace(" ", "_")) for c in categories}
    if category is None and label.replace(" ", "_") in allowed:
        category = label.replace(" ", "_")
    base = {"id": candidate_id, "category": category or label, "confidence": None,
            "source": context["source"], "tile_id": context.get("tile_id"),
            "detection_method": method, "raw_label": raw_label}
    try:
        score = float(item.get("confidence", 0))
        if not math.isfinite(score) or not 0 <= score <= 1:
            return None, {**base, "reason": "invalid_confidence", "raw_confidence": repr(item.get("confidence"))}
        base["confidence"] = score
        box = item.get("bbox")
        fmt = item.get("bbox_format", "xywh" if isinstance(box, dict) else "xyxy")
        values = [box[k] for k in ("x", "y", "w", "h")] if isinstance(box, dict) else list(box)
        if fmt == "xywh":
            x, y, w, h = values
            values = [x, y, x+w, y+h]
        elif fmt != "xyxy":
            raise ValueError("Unsupported bbox format")
        mode = item.get("coordinate_space", "pixel")
        if mode not in {"pixel", "normalized"}:
            raise ValueError("coordinate_space must be pixel or normalized")
        wx, wy, wr, wb = context["window"]
        restored = restore_bbox(values, wx, wy, wr-wx, wb-wy, normalized=mode == "normalized")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return None, {**base, "reason": "invalid_bbox", "raw_bbox": repr(item.get("bbox")), "error": str(error)}
    bbox = (restored["x"], restored["y"], restored["x"]+restored["w"], restored["y"]+restored["h"])
    if category is None:
        return None, {**base, "bbox": bbox, "reason": "category_unknown"}
    edges = truncated_edges(bbox, context["window"], image.size, config.truncation.edge_threshold)
    observation = {**base, "bbox": bbox, "window": context["window"],
                   "prompt_group": context["group"], "is_truncated": bool(edges),
                   "truncated_edges": edges, "parent_id": context.get("parent_id")}
    patch = np.asarray(image.crop(bbox).resize((24, 24)).convert("RGB"))
    hist = np.histogramdd(patch.reshape(-1, 3), bins=(4, 4, 4), range=((0, 256),)*3)[0].ravel()
    hist = (hist / max(hist.sum(), 1)).tolist()
    return DetectionCandidate(
        id=candidate_id, category=category, group=category_group(category),
        subtype=category if "_" in category else None, aliases=[raw_label], confidence=score,
        bbox=bbox, source=context["source"], tile_id=context.get("tile_id"),
        detection_method=method, window=tuple(context["window"]), is_truncated=bool(edges),
        truncated_edges=edges, observations=[observation], appearance=hist,
        parent_id=context.get("parent_id"), redetected=context["source"] == "redetection",
    ), None
