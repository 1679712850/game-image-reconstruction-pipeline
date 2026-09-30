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
from taxonomy.categories import CATEGORY_GROUPS, category_group
from detection.budget import BudgetTracker

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
        scene_categories = list(dict.fromkeys(normalize_category(c, default=c.replace(" ", "_")) for c in categories))
        if cfg.expand_categories:
            categories = [*categories, *(c for group in CATEGORY_GROUPS.values() for c in group)]
        categories = list(dict.fromkeys(normalize_category(c, default=c.replace(" ", "_")) for c in categories))
        scans, raw, invalid, tiles = [], [], [], []
        from services.execution import _active
        runtime = _active.get()
        budget = runtime.detection_budget if runtime is not None else BudgetTracker(cfg.budget)
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
        if cfg.global_detection.enabled and budget.begin_pass("global"):
            detect_global(image, categories, infer, collect, scans,
                          min(cfg.prompt_group_size, cfg.budget.max_categories_per_pass), budget=budget)
        elif cfg.global_detection.enabled:
            scans.append({'source':'global','tile_id':None,'window':[0,0,image.width,image.height],
                          'categories':[],'status':'budget_exhausted','candidates':0,'pass_id':'global_discovery'})
        if cfg.tiling.enabled:
            if budget.begin_pass("tile"):
                tiles = detect_tiles(image, categories, infer, collect, scans, cfg, budget=budget, candidates=raw,
                                     scene_categories=scene_categories,
                                     pass_id='regional_tiled' if round_index == 1 else 'gap_fill')
            else:
                scans.append({'source':'tile','tile_id':None,'window':[0,0,image.width,image.height],
                              'categories':[],'status':'budget_exhausted','candidates':0,'pass_id':'gap_fill'})
        initial, _ = fuse_candidates(raw, cfg.dedup)
        edge_candidates = [c for c in initial if c.is_truncated]
        LOG.info("[Truncation] candidates=%d", len(edge_candidates))
        raw, invalid, redetections = recover_edges(image,edge_candidates,raw,invalid,infer,collect,scans,cfg,budget)
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
            "budget": {**budget.report(), "exhausted":any(s["status"] == "budget_exhausted" for s in scans)},
            "scan_cost": scan_cost(scans,raw,kept),
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


def scan_cost(scans: list[dict], raw: list, kept: list) -> dict:
    """Inference costs by semantic group; shared prompts are explicitly overlapping."""
    result = {}
    for group in sorted({category_group(c) for scan in scans for c in scan.get('categories',[])}):
        related = [scan for scan in scans if any(category_group(c)==group for c in scan.get('categories',[]))]
        result[group] = {
            'inference_calls':sum(s['status'] in {'ok','failed'} for s in related),
            'runtime':sum(s.get('runtime',0) for s in related),
            'candidates':sum(category_group(c.category)==group for c in raw),
            'retained':sum(category_group(c.category)==group for c in kept),
            'budget_skips':sum(s['status']=='budget_exhausted' for s in related)}
    return {'by_group':result,'inference_calls':sum(s['status'] in {'ok','failed'} for s in scans),
            'definition':'Calls/runtime of a shared multi-group prompt occur in each related group; retained is detector filtering, not GT precision.'}


def recover_edges(image, parents: list, raw: list, invalid: list, infer, collect,
                  scans: list[dict], config, budget: BudgetTracker) -> tuple[list,list,list[dict]]:
    """Re-observe only truncated candidates; unrelated discoveries stay auditable."""
    logs = []
    for index,parent in enumerate(parents):
        if index >= config.truncation.max_redetections:
            logs.append({'parent_id':parent.id,'status':'budget_exhausted'})
            continue
        window = expanded_region(parent,image.size,config.truncation.redetect_padding)
        before = len(raw)
        scan_window(image,window,[parent.category],infer,collect,scans,source='redetection',
                    tile_id=f'redetect_{index:04d}',parent_id=parent.id,
                    group_size=min(config.prompt_group_size,config.budget.max_categories_per_pass),
                    budget=budget,pass_id='gap_fill')
        matches = []
        for child in raw[before:]:
            _,overlap,distance,ratio = geometry(parent,child)
            if child.category == parent.category and overlap >= config.dedup.overlap_ratio and distance <= 1.25 and ratio >= .1:
                matches.append(child)
            else:
                invalid.append({**child.serializable(),'reason':'redetection_unmatched'})
        del raw[before:]
        raw.extend(matches)
        logs.append({'parent_id':parent.id,'window':window,
                     'status':'recovered' if any(not c.is_truncated for c in matches) else 'unresolved'})
    return raw,invalid,logs
