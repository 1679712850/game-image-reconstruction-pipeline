"""High-recall scheduling, candidate fusion and bounded edge recovery."""
import logging
from collections import Counter

from detection.candidate_adapter import observation_candidate
from detection.global_detector import detect_global
from detection.grouped_detector import scan_window
from detection.tile_detector import detect_tiles
from fusion.candidate_fusion import fuse_candidates
from fusion.cross_tile_dedup import geometry
from postprocess.candidate_filter import filter_candidates
from postprocess.small_object_filter import threshold_for
from postprocess.truncation_detector import expanded_region
from taxonomy.aliases import normalize_category
from taxonomy.categories import CATEGORY_GROUPS

LOG = logging.getLogger(__name__)


def scene_record(candidate):
    return {"category": candidate.category, "confidence": candidate.confidence,
            "bbox": candidate.as_bbox(), "group": candidate.group, "subtype": candidate.subtype,
            "aliases": candidate.aliases, "source": candidate.source, "tile_id": candidate.tile_id,
            "is_truncated": candidate.is_truncated, "truncated_edges": candidate.truncated_edges,
            "detection_method": candidate.detection_method, "parent_id": candidate.parent_id,
            "source_candidates": [o["id"] for o in candidate.observations],
            "merged_from": candidate.merged_from, "merged": len(candidate.observations) > 1,
            "observations": candidate.observations, "redetected": candidate.redetected,
            "review_required": candidate.review_required, "review_reason": candidate.review_reason}


class P0DetectionPipeline:
    def __init__(self, config):
        self.config = config

    def run(self, image, categories, infer, *, method="grounding", round_index=1):
        cfg = self.config
        if cfg.expand_categories:
            categories = [*categories, *(c for group in CATEGORY_GROUPS.values() for c in group)]
        categories = list(dict.fromkeys(normalize_category(c, default=c.replace(" ", "_")) for c in categories))
        scans, raw, invalid, tiles = [], [], [], []
        serial = 0
        def collect(item, context):
            nonlocal serial
            serial += 1
            candidate_id = f"r{round_index:02d}_{context.get('tile_id') or context['source']}_candidate_{serial:06d}"
            candidate, error = observation_candidate(item, context, image, candidate_id, cfg, method, categories)
            if error:
                invalid.append(error)
            else:
                raw.append(candidate)
                LOG.debug("[CoordinateRestore] %s bbox=%s", candidate_id, candidate.bbox)
        if cfg.global_detection.enabled:
            detect_global(image, categories, infer, collect, scans, cfg.prompt_group_size)
        if cfg.tiling.enabled:
            tiles = detect_tiles(image, categories, infer, collect, scans, cfg)
        initial, _ = fuse_candidates(raw, cfg.dedup)
        edge_candidates = [c for c in initial if c.is_truncated]
        LOG.info("[Truncation] candidates=%d", len(edge_candidates))
        redetections = []
        for index, parent in enumerate(edge_candidates):
            if index >= cfg.truncation.max_redetections:
                redetections.append({"parent_id": parent.id, "status": "budget_exhausted"})
                continue
            window = expanded_region(parent, image.size, cfg.truncation.redetect_padding)
            before = len(raw)
            LOG.info("[Redetection] parent=%s window=%s", parent.id, window)
            scan_window(image, window, [parent.category], infer, collect, scans, source="redetection",
                        tile_id=f"redetect_{index:04d}", parent_id=parent.id, group_size=cfg.prompt_group_size)
            matches = []
            for child in raw[before:]:
                _, overlap, distance, ratio = geometry(parent, child)
                if child.category == parent.category and overlap >= cfg.dedup.overlap_ratio and distance <= 1.25 and ratio >= .1:
                    matches.append(child)
                else:
                    invalid.append({**child.serializable(), "reason": "redetection_unmatched"})
            del raw[before:]
            raw.extend(matches)
            redetections.append({"parent_id": parent.id, "window": window,
                                 "status": "recovered" if any(not c.is_truncated for c in matches) else "unresolved"})
        fused, duplicates = fuse_candidates(raw, cfg.dedup)
        kept, rejected, review = filter_candidates(fused, image.width*image.height, cfg.confidence)
        rejected_records = [*invalid, *({**c.serializable(), "reason": c.reject_reason} for c in [*duplicates, *rejected])]
        small = [c for c in kept if c.area < image.width*image.height*cfg.confidence.small_area_ratio]
        tile_ids = {t["tile_id"] for t in tiles}
        failed_tiles = sorted({s["tile_id"] for s in scans if s["status"] == "failed" and s["tile_id"] in tile_ids})
        stats = {
            "round": round_index, "backend": method, "bbox_format": "absolute_pixel_xyxy",
            "small_area_ratio": cfg.confidence.small_area_ratio,
            "global_candidates": sum(c.source == "global" for c in raw)+sum(c.get("source") == "global" for c in invalid),
            "tile_candidates": sum(c.source == "tile" for c in raw)+sum(c.get("source") == "tile" for c in invalid),
            "redetection_candidates": sum(c.source == "redetection" for c in raw)+sum(c.get("source") == "redetection" for c in invalid),
            "combined_candidates": len(raw)+len(invalid), "after_dedup": len(fused),
            "after_filter": len(kept), "category_counts": dict(Counter(c.category for c in kept)),
            "failed_tiles": failed_tiles, "tile_count": len(tiles),
            "scans": scans, "tiles": tiles, "redetections": redetections,
            "candidates": [*[c.serializable() for c in raw], *invalid], "merged": [c.serializable() for c in fused],
            "selected_candidates": [c.serializable() for c in kept], "filtered": rejected_records,
            "review_candidate_pool": [c.serializable() for c in review],
            "small_object_report": {
                "small_object_count": len(small),
                "detected_by_global": sum(any(o["source"] == "global" for o in c.observations) for c in small),
                "detected_by_tile": sum(any(o["source"] == "tile" for o in c.observations) for c in small),
                "tile_only": sum(not any(o["source"] == "global" for o in c.observations) for c in small),
                "definition": "Unique retained boxes below configured image-area ratio; source counts may overlap, not semantic recall",
            },
        }
        LOG.info("[Dedup] before=%d after=%d", len(raw), len(fused))
        LOG.info("[Filter] kept=%d rejected=%d review=%d", len(kept), len(rejected_records), len(review))
        records = [dict(scene_record(c), confidence_threshold=threshold_for(c, image.width*image.height, cfg.confidence)) for c in kept]
        return records, stats
