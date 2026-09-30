"""Build a portable, validated UTF-8 scene manifest."""
import json
from pathlib import Path

from agent.state import SceneState
from app.paths import relative_asset
from schemas.scene import ExportObject, SceneManifest
from diagnostics.p1_report import portable


def build_manifest(state: SceneState, mock: bool, backends: dict[str, str] | None = None) -> SceneManifest:
    """Replace internal absolute paths with paths relative to scene.json."""
    root = Path(state["output_dir"])
    objects = []
    for record in state.get("objects", []):
        data = dict(record)
        for key in ("asset_path", "hd_asset_path", "mask_path", "candidate_mask_path", "visible_mask_path", "full_mask_path"):
            data[key] = relative_asset(data.get(key), root)
        data.update(asset=data["asset_path"], hd_asset=data["hd_asset_path"])
        objects.append(ExportObject.model_validate(portable(data, root)))
    analysis = state["scene_analysis"]
    preview = state.get("reconstruction_path")
    layers, edits = [], []
    for record in state.get("decomposed_layers", []):
        layers.append({**record, "asset_path": relative_asset(record["asset_path"], root)})
    for record in state.get("object_edits", []):
        data = dict(record)
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
        detection={"rounds": [{key: run.get(key) for key in ("round", "global_candidates", "tile_candidates", "combined_candidates", "after_dedup", "after_filter", "failed_tiles", "small_object_report")} for run in state.get("detection_runs", [])],
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
        p1_summary=portable(state.get('p1_summary', {}), root),
    )


def export_json(manifest: SceneManifest, path: str | Path) -> str:
    """Validate before writing; preserve non-ASCII labels and disallow NaN."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(target.resolve())
