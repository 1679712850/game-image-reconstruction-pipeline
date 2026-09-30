"""Build a portable, validated UTF-8 scene manifest."""
import json
from pathlib import Path

from agent.state import SceneState
from app.paths import relative_asset
from schemas.scene import ExportObject, SceneManifest
from diagnostics.p1_report import portable



def export_geometry(record: dict, root: Path) -> dict:
    """Source-visible bounds and final accepted placement, never an unaccepted proposal."""
    visible = record.get('bbox_visible') or record.get('bbox')
    mask = record.get('visible_mask_path') or record.get('source_mask_path')
    if mask and (root/mask).is_file():
        from PIL import Image
        with Image.open(root/mask) as image:
            bounds = image.convert('L').point(lambda value: 255 if value > 8 else 0).getbbox()
        visible = {'x':bounds[0],'y':bounds[1],'w':bounds[2]-bounds[0],'h':bounds[3]-bounds[1]} if bounds else None
    return {'visible_bbox':visible,'visible_mask':mask,
            'full_asset_bbox':record.get('placement',{}).get('crop_bbox') or record.get('crop_bbox'),
            'full_asset_canvas':record.get('logical_size'),
            'full_asset_mask':record.get('reconstructed_mask_path')}

def build_manifest(state: SceneState, mock: bool, backends: dict[str, str] | None = None) -> SceneManifest:
    """Replace internal absolute paths with paths relative to scene.json."""
    root = Path(state["output_dir"])
    objects = []
    for record in state.get("objects", []):
        data = dict(record)
        geometry = export_geometry(data, root)
        for key in ("asset_path", "hd_asset_path", "mask_path", "candidate_mask_path", "visible_mask_path", "full_mask_path"):
            data[key] = relative_asset(data.get(key), root)
        if data.get("accepted_asset"):
            data["asset_path"] = relative_asset(data["accepted_asset"], root)
        data.update(asset=data["asset_path"], hd_asset=data["hd_asset_path"], geometry=geometry)
        objects.append(ExportObject.model_validate(portable(data, root)))
    analysis = state["scene_analysis"]
    preview = state.get("reconstruction_path")
    layers, edits = [], []
    for record in state.get("decomposed_layers", []):
        layers.append({**record, "asset_path": relative_asset(record["asset_path"], root)})
    for record in state.get("object_edits", []):
        data = dict(record)
        # Older checkpoints may carry the retired label; normalize metadata while
        # keeping the actual candidate alpha/resegmentation contract current.
        if data.get('alpha_policy') == 'preserve_source':
            data['alpha_policy'] = 'visible_hint'
        for key in ("asset_path", "source_asset_path", "mask_path"):
            data[key] = relative_asset(data[key], root)
        edits.append(data)
    return SceneManifest(
        mock=mock,
        backends=backends or {},
        scene={"width": state["width"], "height": state["height"], "projection": analysis["projection"]},
        description=analysis["description"], layers=state.get("layer_plan", []),
        decomposed_layers=layers, object_edits=edits,
        objects=[obj for obj in objects if obj.group != "fx_environment"],
        environment_effects=[obj for obj in objects if obj.group == "fx_environment"],
        detection={"instances": portable(state.get('all_detections', state.get('detections', [])), root), "object_limit": {
            "truncated": any(run.get('object_limit',{}).get('truncated',False) for run in state.get('detection_runs',[])),
            "objects_before_limit":sum(run.get('object_limit',{}).get('objects_before_limit',0) for run in state.get('detection_runs',[])),
            "objects_after_limit":sum(run.get('object_limit',{}).get('objects_after_limit',0) for run in state.get('detection_runs',[])),
        }, "budget": state.get("detection_budget", {}), "rounds": [{key: run.get(key) for key in ("budget", "scan_cost", "round", "global_candidates", "tile_candidates", "combined_candidates", "after_dedup", "after_filter", "failed_tiles", "small_object_report")} for run in state.get("detection_runs", [])],
                   "review_candidate_pool": [c for run in state.get("detection_runs", []) for c in run.get("review_candidate_pool", [])]},
        retry_count=state.get("retry_count", 0),
        retry_history=portable(state.get("retry_history", []), root),
        unresolved_objects=state.get("failed_objects", []),
        reconstruction=relative_asset(preview, root) if preview else None,
        reconstruction_score=state.get("reconstruction_score") if preview else None,
        coverage={"accepted": state.get("scene_coverage", 0.0),
                  "candidate": state.get("scene_qa", {}).get("candidate_coverage", 0.0),
                  "remaining_image": relative_asset(state.get("working_path"), root) if state.get("scene_history") else None,
                  "mask": relative_asset(state.get("coverage_mask_path") or None, root),
                  "definition": "Union of masks over nontransparent source pixels; not semantic recall"},
        scene_qa={"rounds": state.get("scene_history", []), "stop_reason": state.get("scene_stop_reason", "")},
        ownership=portable(state.get("ownership", {}), root),
        terrain_layers=portable(state.get('terrain_layers', []), root),
        accepted_ownership=portable(state.get("accepted_ownership", {}), root),
        resource_failures=state.get("resource_failures", []),
        candidate_registry=portable(state.get("candidate_registry", {}), root),
        psd=relative_asset(state.get("psd_path"), root),
        p1_summary=portable(state.get('p1_summary', {}), root),
        completion_metrics=portable(state.get('completion_metrics', {}), root),
        pipeline_status=state.get('pipeline_status', 'completed'),
    )


def export_json(manifest: SceneManifest, path: str | Path) -> str:
    """Validate before writing; preserve non-ASCII labels and disallow NaN."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(target.resolve())
