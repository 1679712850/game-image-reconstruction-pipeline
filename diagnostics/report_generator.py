"""Complete, portable diagnostics across all rounds and segmentation outcomes."""
import html
import json
import logging
from collections import Counter
from pathlib import Path
from diagnostics.bbox_visualizer import draw_candidates
from diagnostics.coverage_map import draw_coverage
from diagnostics.tile_visualizer import draw_tiles


IMAGES = ("global_detection", "tile_detection", "merged_detection", "filtered_detection", "coverage_map", "tile_grid")


def generate_report(source, directory, runs, objects=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [c for run in runs for c in run.get("candidates", [])]
    rejected = [c for run in runs for c in run.get("filtered", [])]
    review = [c for run in runs for c in run.get("review_candidate_pool", [])]
    merged = [c for run in runs for c in run.get("merged", [])]
    selected = objects if objects is not None else [c for run in runs for c in run.get("selected_objects", [])]
    scans = [s for run in runs for s in run.get("scans", [])]
    tiles = list({t["tile_id"]: t for run in runs for t in run.get("tiles", [])}.values())
    failures = [{"round": run.get("round"), **s} for run in runs for s in run.get("scans", []) if s.get("status") == "failed"]
    for obj in selected:
        if obj.get("error") or (obj.get("status") in {"manual_review", "retry"} and not obj.get("asset_path")):
            rejected.append({"id": obj["id"], "category": obj["category"], "confidence": obj["confidence"],
                             "source_candidates": obj.get("source_candidates", []),
                             "reason": "mask_too_small" if obj.get("error") == "empty mask" else "segmentation_failed",
                             "error": obj.get("error")})
    draw_candidates(source, directory / "global_detection.png", [c for c in candidates if c.get("source") == "global"])
    draw_candidates(source, directory / "tile_detection.png", [c for c in candidates if c.get("source") == "tile"])
    draw_candidates(source, directory / "merged_detection.png", merged)
    draw_candidates(source, directory / "filtered_detection.png", selected)
    draw_tiles(source, directory / "tile_grid.png", tiles, [s["tile_id"] for s in failures])
    coverage = draw_coverage(source, directory / "coverage_map.png", scans, tiles, candidates)
    summary = {key: sum(run.get(key, 0) for run in runs) for key in
               ("global_candidates", "tile_candidates", "redetection_candidates", "combined_candidates", "after_dedup", "after_filter")}
    summary.update(final_objects=len(selected), category_counts=dict(Counter(c["category"] for c in selected)),
                   rejection_reasons=dict(Counter(c["reason"] for c in rejected)),
                   failed_tiles=len({(s["round"], s["tile_id"]) for s in failures if s["source"] == "tile"}),
                   total_tiles=sum(run.get("tile_count", 0) for run in runs), failed_scans=failures,
                   coverage=coverage, rounds=len(runs), review_candidates=len(review),
                   bbox_format="candidates: absolute pixel xyxy; scene objects: absolute pixel xywh",
                   count_definition="Detection counters sum observations per round; final_objects are unique exported records")
    small = {"per_round": [r.get("small_object_report", {}) for r in runs],
             "definition": "Diagnostic counts, not measured recall. Global/tile source counts can overlap."}
    if objects is not None:
        # Final records carry their own scale-dependent confidence threshold and provenance.
        from PIL import Image
        with Image.open(source) as original:
            area = original.width * original.height
        limit = runs[0].get("small_area_ratio", .001) if runs else .001
        small_objects = [o for o in objects if o["bbox"]["w"]*o["bbox"]["h"] < area*limit]
        small.update(small_object_count=len(small_objects),
                     detected_by_global=sum(any(v["source"] == "global" for v in o.get("observations", [])) for o in small_objects),
                     detected_by_tile=sum(any(v["source"] == "tile" for v in o.get("observations", [])) for o in small_objects))
    for name, values in (("candidates", candidates), ("filtered", rejected), ("review_candidate_pool", review),
                         ("summary", summary), ("small_object_report", small), ("scans", scans), ("objects", selected)):
        (directory / f"{name}.json").write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    escaped = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    figures = "".join(f'<figure><figcaption>{name}</figcaption><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></figure>' for name in IMAGES)
    links = " | ".join(f'<a href="{name}.json">{name}.json</a>' for name in ("candidates", "filtered", "review_candidate_pool", "small_object_report", "scans", "objects"))
    (directory / "report.html").write_text(f'<!doctype html><meta charset="utf-8"><title>Detection diagnostics</title><style>body{{font:15px system-ui;background:#17202d;color:#eee;margin:24px}}a{{color:#81dbff}}img{{max-width:100%}}pre{{white-space:pre-wrap}}figure{{margin:24px 0}}</style><h1>Detection diagnostics</h1><p>Failed Tiles: {summary["failed_tiles"]} / {summary["total_tiles"]}</p><p>{links}</p><p>Blue regions were scanned but have no candidates. Red regions were not scanned. Coverage is not semantic recall.</p><pre>{escaped}</pre>{figures}', encoding="utf-8")
    logging.getLogger(__name__).info("[Diagnostics] %s", directory / "report.html")
    return summary
