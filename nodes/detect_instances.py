"""Detection node with stable IDs, independent from model initialization."""
from collections import Counter
from collections.abc import Callable
import re
import json
from pathlib import Path

from app.config import SceneLoopConfig
from app.detection_config import DetectionConfig
from cv.mask import read_mask
from services.detection_postprocess import iou

from agent.state import SceneState
from schemas.object import SceneObject
from services.grounding_service import GroundingService


def make_detect_instances(service: GroundingService, exercise_retry: bool = False, loop: SceneLoopConfig | None = None, detection: DetectionConfig | None = None) -> Callable[[SceneState], dict]:
    """Bind a detector and optionally inject one demonstrable mock failure."""
    def detect_instances(state: SceneState) -> dict:
        """Return schema-validated detections in source-image coordinates."""
        categories = list(dict.fromkeys(
            category for layer in state["layer_plan"] for category in layer["categories"]
        ))
        layers = [{**layer, "categories": list(layer["categories"])} for layer in state["layer_plan"]]
        additional = [c for c in state.get("scene_next_categories", []) if c not in categories]
        if additional:
            other = next((layer for layer in layers if layer["name"] == "discovered"), None)
            if other is None:
                layers.append({"name": "discovered", "categories": additional, "order": len(layers)})
            else:
                other["categories"].extend(additional)
        categories = list(dict.fromkeys([*state.get("scene_next_categories", []), *categories]))
        round_index = state.get("detection_round", 0) + 1
        path = state.get("working_path", state["source_path"])
        if detection is not None and type(service).detect_round is GroundingService.detect_round:
            raw, diagnostics = service.detect_p0(path, categories, detection, round_index)
        elif loop is not None:
            raw, diagnostics = service.detect_round(path, categories, round_index)
        else:
            raw, diagnostics = service.detect(path, categories), {}
        archive = state.get("archived_objects", [])
        coverage = read_mask(state["coverage_mask_path"]) > 0 if state.get("coverage_mask_path") else None
        counts, detections = Counter(), []
        used = {obj["id"] for obj in archive}
        assigned = set()
        dropped = 0
        filtered_records = diagnostics.setdefault("filtered", [])
        def reject(item, reason):
            box = item["bbox"]
            filtered_records.append({**item, "bbox": [box["x"], box["y"], box["x"]+box["w"], box["y"]+box["h"]],
                                     "reason": reason, "id": (item.get("source_candidates") or [f"r{round_index}_external_{len(filtered_records)}"])[0]})
        for index, item in enumerate(raw):
            box = item["bbox"]
            x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
            if coverage is not None and coverage[y:y+h, x:x+w].mean() >= loop.covered_box_threshold:
                dropped += 1
                reject(item, "covered_by_accepted_mask")
                continue
            matches = [o for o in archive if o["category"] == item["category"] and iou(o["bbox"], box) > service.config.grounding.nms_iou]
            if any(o["status"] == "pass" for o in matches):
                dropped += 1
                reject(item, "duplicate_archived")
                continue
            old = max(matches, key=lambda o: iou(o["bbox"], box), default=None)
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", item["category"]).strip("_") or "object"
            if old is not None:
                object_id = old["id"]
            else:
                counts[slug] += 1
                object_id = f"{slug}_{counts[slug]:03d}"
                while object_id in used:
                    counts[slug] += 1
                    object_id = f"{slug}_{counts[slug]:03d}"
                if loop is not None and len(used) >= loop.max_objects:
                    dropped += 1
                    reject(item, "max_objects")
                    continue
            if object_id in assigned:
                reject(item, "duplicate_assigned")
                continue
            used.add(object_id)
            assigned.add(object_id)
            obj = SceneObject.model_validate({**item, "id": object_id,
                "confidence": 0.1 if exercise_retry and index == 0 else item["confidence"]})
            if obj.bbox.x + obj.bbox.w > state["width"] or obj.bbox.y + obj.bbox.h > state["height"]:
                raise ValueError(f"Detection bbox lies outside source: {obj.id}")
            detections.append(obj.model_dump(mode="json"))
            layer = next((layer for layer in layers if obj.category in layer["categories"]), None)
            if layer is None:
                layer = next((layer for layer in layers if layer["name"] == obj.group), None)
                if layer is None:
                    layer = {"name": obj.group, "categories": [], "order": len(layers)}
                    layers.append(layer)
                layer["categories"].append(obj.category)
        diagnostics.update(round=round_index, categories=categories, removed_or_duplicate=dropped, selected=len(detections), selected_objects=detections)
        runs = [*state.get("detection_runs", []), diagnostics]
        if detection is not None and detection.diagnostics.enabled:
            from diagnostics.report_generator import generate_report
            generate_report(state["source_path"], Path(state["output_dir"]) / "diagnostics", runs)
        if loop is not None:
            directory = Path(state["output_dir"]) / "debug" / f"round_{round_index:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "detections.json").write_text(json.dumps({**diagnostics, "detections": detections}, ensure_ascii=False, indent=2), encoding="utf-8")
        all_detections = {obj["id"]: obj for obj in state.get("all_detections", [])}
        all_detections.update({obj["id"]: obj for obj in detections})
        return {"detections": detections, "all_detections": list(all_detections.values()), "detection_round": round_index, "retry_count": 0,
                "failed_objects": [], "detection_diagnostics": diagnostics, "detection_runs": runs, "layer_plan": layers}
    return detect_instances
