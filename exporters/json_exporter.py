"""Build a portable, validated UTF-8 scene manifest."""
import json
from pathlib import Path

from agent.state import SceneState
from app.paths import relative_asset
from schemas.scene import ExportObject, SceneManifest


def build_manifest(state: SceneState, mock: bool) -> SceneManifest:
    """Replace internal absolute paths with paths relative to scene.json."""
    root = Path(state["output_dir"])
    objects = []
    for record in state.get("objects", []):
        data = dict(record)
        for key in ("asset_path", "hd_asset_path", "mask_path"):
            data[key] = relative_asset(data.get(key), root)
        data.update(asset=data["asset_path"], hd_asset=data["hd_asset_path"])
        objects.append(ExportObject.model_validate(data))
    analysis = state["scene_analysis"]
    preview = state.get("reconstruction_path")
    return SceneManifest(
        mock=mock,
        scene={"width": state["width"], "height": state["height"], "projection": analysis["projection"]},
        description=analysis["description"], layers=state.get("layer_plan", []),
        objects=objects, retry_count=state.get("retry_count", 0),
        unresolved_objects=state.get("failed_objects", []),
        reconstruction=relative_asset(preview, root) if preview else None,
        reconstruction_score=state.get("reconstruction_score") if preview else None,
    )


def export_json(manifest: SceneManifest, path: str | Path) -> str:
    """Validate before writing; preserve non-ASCII labels and disallow NaN."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(target.resolve())
