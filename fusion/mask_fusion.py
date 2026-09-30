"""Fuse full-source masks only with class, spatial and mask-overlap evidence."""
import numpy as np
from cv.mask import read_mask, save_mask
from services.detection_postprocess import iou


def mask_iou(a, b):
    aa, bb = np.asarray(a) > 0, np.asarray(b) > 0
    if aa.shape != bb.shape:
        raise ValueError("Mask fusion requires the same absolute source coordinate frame")
    union = np.count_nonzero(aa | bb)
    return np.count_nonzero(aa & bb) / union if union else 0.0


def fuse_segmented_records(records, config):
    kept, rejected = [], []
    for original in sorted(records, key=lambda r: r["confidence"], reverse=True):
        record = dict(original)
        mask = read_mask(record["mask_path"]) if record.get("mask_path") else None
        match = None
        if mask is not None and np.any(mask):
            for index, other in enumerate(kept):
                if record["category"] != other["category"] or not other.get("mask_path"):
                    continue
                if iou(record["bbox"], other["bbox"]) <= 0:
                    continue
                other_mask = read_mask(other["mask_path"])
                if mask_iou(mask, other_mask) >= config.mask_iou:
                    match = index
                    break
        if match is None:
            kept.append(record)
            continue
        other = kept[match]
        save_mask(np.maximum(mask, other_mask), other["mask_path"])
        a, b = other["bbox"], record["bbox"]
        x, y = min(a["x"], b["x"]), min(a["y"], b["y"])
        other["bbox"] = {"x": x, "y": y, "w": max(a["x"]+a["w"], b["x"]+b["w"])-x,
                         "h": max(a["y"]+a["h"], b["y"]+b["h"])-y}
        other["source_candidates"] = list(dict.fromkeys([*other.get("source_candidates", []), *record.get("source_candidates", [])]))
        other["observations"] = list({o["id"]: o for o in [*other.get("observations", []), *record.get("observations", [])]}.values())
        other["merged_from"] = list(dict.fromkeys([*other.get("merged_from", []), record["id"], *record.get("merged_from", [])]))
        other["merged"] = True
        other["redetected"] = other.get("redetected", False) or record.get("redetected", False)
        rejected.append({"id": record["id"], "category": record["category"], "confidence": record["confidence"],
                         "source_candidates": record.get("source_candidates", []), "parent_id": other["id"], "reason": "mask_duplicate"})
    # Preserve stable caller ordering for retries, manifests and mock injection.
    order = {r["id"]: i for i, r in enumerate(records)}
    return sorted(kept, key=lambda r: order[r["id"]]), rejected
