"""Model-independent label normalization, clipping and class-aware NMS."""
import math
import re


def normalize_label(label: str, categories: list[str]) -> str | None:
    """Match phrases to configured labels; reject unknown/ambiguous phrases."""
    label = re.sub(r"\s+", " ", label.strip().lower().strip(". "))
    exact = [c for c in categories if c.lower() == label]
    if exact:
        return exact[0]
    matches = [c for c in categories if re.search(r"(?<!\w)" + re.escape(c.lower()) + r"(?!\w)", label)]
    return matches[0] if len(matches) == 1 else None


def iou(a: dict, b: dict) -> float:
    """Intersection over union of two integer xywh boxes."""
    x0, y0 = max(a['x'], b['x']), max(a['y'], b['y'])
    x1, y1 = min(a['x'] + a['w'], b['x'] + b['w']), min(a['y'] + a['h'], b['y'] + b['h'])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    return overlap / (a['w'] * a['h'] + b['w'] * b['h'] - overlap)


def postprocess_detections(
    boxes: list[list[float]], scores: list[float], labels: list[str],
    categories: list[str], width: int, height: int,
    threshold: float, nms_iou: float, limit: int,
) -> list[dict]:
    """Convert absolute xyxy to safe xywh, then keep top nonduplicate boxes."""
    if not len(boxes) == len(scores) == len(labels):
        raise ValueError('Detector returned mismatched boxes/scores/labels')
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) != 4 or not all(math.isfinite(v) for v in [*box, score]):
            continue
        category = normalize_label(label, categories)
        if category is None or not threshold <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        x0, y0 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
        x1, y1 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
        if x1 <= x0 or y1 <= y0:
            continue
        candidates.append({'category': category, 'confidence': float(score), 'bbox': {'x': x0, 'y': y0, 'w': x1-x0, 'h': y1-y0}})
    kept = []
    for item in sorted(candidates, key=lambda x: x['confidence'], reverse=True):
        if any(item['category'] == other['category'] and iou(item['bbox'], other['bbox']) > nms_iou for other in kept):
            continue
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept
